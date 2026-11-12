// Behavioral row partition. Arithmetic remains SystemVerilog, not host BLAS.
// Scalar and bulk-row transports are selectable; DPI never performs model arithmetic.
module projection_worker (
    input logic clk, start,
    input longint base,
`ifdef PARALLEL_ATTENTION
    input int operation, attention_layer, attention_position,
`endif
`ifdef ATTENTION_QUERY_VALUE_DISPATCH
    input int query_value_min_context,
    output longint unsigned query_value_paired_count, query_value_fallback_count,
`endif
    input int first_row, end_row, width, reduction_lanes,
`ifdef BATCH_PREFILL
    input int batch_count,
`ifdef WIDE_BATCH
    // Probe-only ports: no public model/block protocol is changed.
    input real wide_activation[4096*`WIDE_BATCH],
    output real wide_result[3072*`WIDE_BATCH],
`endif
`ifdef BLOCK_UNROLL
    input real batch_activation[16384],
`else
    input real batch_activation[12288],
`endif
    output real batch_result[12288],
`endif
`ifdef MASK_ACTIVATION_INDEX
    input real activation[4096],
`else
    input real activation[3072],
`endif
    output real result[151936]
);
`ifdef UNCHECKED_WEIGHTS
    import "DPI-C" function real weight_value_unchecked(input longint index);
    function automatic real projection_weight_value(input longint index);
        return weight_value_unchecked(index);
    endfunction
`else
    import "DPI-C" function real weight_value(input longint index);
    function automatic real projection_weight_value(input longint index);
        return weight_value(index);
    endfunction
`endif
`ifdef BULK_ROW_WEIGHTS
    import "DPI-C" function void weight_row(input longint index, input int count,
                                            output real values[]);
    real row_weights[3072];
`endif

`ifdef MASK_ACTIVATION_INDEX
`define ACTIVATION_AT(index) activation[(index)&4095]
`else
`define ACTIVATION_AT(index) activation[index]
`endif
`ifdef PARALLEL_ATTENTION
    import "DPI-C" function real attention_key_value(input int index);
    import "DPI-C" function real attention_value_value(input int index);
    real attention_scores[4096];
`ifdef ATTENTION_HEAD_PAIRS
`ifdef ATTENTION_DOT_LANES8
    initial $fatal(1,"Paired-head attention requires the original serial QK reduction");
`endif
    // Persistent scratch; no per-dispatch clearing of the context-sized array.
    real head_pair_queries[128][2], head_pair_scores[4096][2];
`ifdef ATTENTION_VALUE_PAIRS
    real head_pair_accum[128][2];
`endif
    task automatic attend_head_pairs;
        real sums[2], maximum[2], key, denom;
`ifdef ATTENTION_VALUE_PAIRS
        real denoms[2], scales[2], value;
`else
        real scale;
`endif
        int index;
        for (int pair=first_row;pair<end_row;pair++) begin
            for (int j=0;j<128;j++) begin
                head_pair_queries[j][0]=`ACTIVATION_AT(pair*256+j);
                head_pair_queries[j][1]=`ACTIVATION_AT(pair*256+128+j);
            end
            maximum[0]=-1.0e300;maximum[1]=-1.0e300;
            for (int t=0;t<=attention_position;t++) begin
                sums[0]=0.0;sums[1]=0.0;
                index=(attention_layer*4096+t)*1024+pair*128;
                for (int j=0;j<128;j++) begin
                    key=attention_key_value(index+j);
                    sums[0]+=head_pair_queries[j][0]*key;
                    sums[1]+=head_pair_queries[j][1]*key;
                end
                head_pair_scores[t][0]=sums[0]/$sqrt(128.0);
                head_pair_scores[t][1]=sums[1]/$sqrt(128.0);
                if(head_pair_scores[t][0]>maximum[0])maximum[0]=head_pair_scores[t][0];
                if(head_pair_scores[t][1]>maximum[1])maximum[1]=head_pair_scores[t][1];
            end
            for (int lane=0;lane<2;lane++) begin
                denom=0.0;
                for (int t=0;t<=attention_position;t++) begin
                    head_pair_scores[t][lane]=$exp(head_pair_scores[t][lane]-maximum[lane]);
                    denom+=head_pair_scores[t][lane];
                end
`ifdef ATTENTION_VALUE_PAIRS
                denoms[lane]=denom;
`else
                for (int j=0;j<128;j++)result[pair*256+lane*128+j]=0.0;
                for (int t=0;t<=attention_position;t++) begin
                    scale=head_pair_scores[t][lane]/denom;
                    index=(attention_layer*4096+t)*1024+pair*128;
                    for (int j=0;j<128;j++)
                        result[pair*256+lane*128+j]+=scale*attention_value_value(index+j);
                end
`endif
            end
`ifdef ATTENTION_VALUE_PAIRS
            for (int j=0;j<128;j++) begin
                head_pair_accum[j][0]=0.0;head_pair_accum[j][1]=0.0;
            end
            // Each output retains its original ascending-t FMA order.
            for (int t=0;t<=attention_position;t++) begin
                scales[0]=head_pair_scores[t][0]/denoms[0];
                scales[1]=head_pair_scores[t][1]/denoms[1];
                index=(attention_layer*4096+t)*1024+pair*128;
                for (int unsigned j=0;j<128;j++) begin
                    value=attention_value_value(index+int'(j));
                    head_pair_accum[j][0]+=scales[0]*value;
                    head_pair_accum[j][1]+=scales[1]*value;
                end
            end
            for (int j=0;j<128;j++) begin
                result[pair*256+j]=head_pair_accum[j][0];
                result[pair*256+128+j]=head_pair_accum[j][1];
            end
`endif
        end
    endtask
`endif
    task automatic attend_heads;
        real sum, maximum, denom, scale;
        int index, kvhead;
`ifdef ATTENTION_DOT_LANES8
        real dot_partial[8];
`endif
        for (int head=first_row;head<end_row;head++) begin
            kvhead=head/2;
            maximum=-1.0e300;
            for (int t=0;t<=attention_position;t++) begin
                sum=0.0;
                index=(attention_layer*4096+t)*1024+kvhead*128;
`ifdef ATTENTION_DOT_LANES8
                for (int lane=0;lane<8;lane++) dot_partial[lane]=0.0;
                for (int unsigned j=0;j<128;j+=8) begin
                    dot_partial[0]+=`ACTIVATION_AT(head*128+j+0)*attention_key_value(index+j+0);
                    dot_partial[1]+=`ACTIVATION_AT(head*128+j+1)*attention_key_value(index+j+1);
                    dot_partial[2]+=`ACTIVATION_AT(head*128+j+2)*attention_key_value(index+j+2);
                    dot_partial[3]+=`ACTIVATION_AT(head*128+j+3)*attention_key_value(index+j+3);
                    dot_partial[4]+=`ACTIVATION_AT(head*128+j+4)*attention_key_value(index+j+4);
                    dot_partial[5]+=`ACTIVATION_AT(head*128+j+5)*attention_key_value(index+j+5);
                    dot_partial[6]+=`ACTIVATION_AT(head*128+j+6)*attention_key_value(index+j+6);
                    dot_partial[7]+=`ACTIVATION_AT(head*128+j+7)*attention_key_value(index+j+7);
                end
                for (int lane=0;lane<8;lane++) sum+=dot_partial[lane];
`else
                for (int j=0;j<128;j++)
                    sum+=`ACTIVATION_AT(head*128+j)*attention_key_value(index+j);
`endif
                attention_scores[t]=sum/$sqrt(128.0);
                if (attention_scores[t]>maximum) maximum=attention_scores[t];
            end
            denom=0.0;
            for (int t=0;t<=attention_position;t++) begin
                attention_scores[t]=$exp(attention_scores[t]-maximum);
                denom+=attention_scores[t];
            end
            for (int j=0;j<128;j++) result[head*128+j]=0.0;
            for (int t=0;t<=attention_position;t++) begin
                scale=attention_scores[t]/denom;
                index=(attention_layer*4096+t)*1024+kvhead*128;
                for (int j=0;j<128;j++)
                    result[head*128+j]+=scale*attention_value_value(index+j);
            end
        end
    endtask
`ifdef BATCH_ATTENTION
`ifdef ATTENTION_QK2
    real pair_scores[4096][2];
`ifdef ATTENTION_QUERY_VALUE_PAIRS
    real query_pair_values[128][2];
`ifdef ATTENTION_QUERY_VALUE_DISPATCH
    initial begin query_value_paired_count=0;query_value_fallback_count=0;end
`endif
`endif
    // Independent queries share each key load. Each query keeps ascending-j
    // accumulation and ascending-t softmax/value order; no horizontal reduction.
    task automatic attend_block_heads_qk2;
        real sums[2], maximum[2], denom, scale, key;
`ifdef ATTENTION_QUERY_VALUE_PAIRS
        real denoms[2], scales[2], value;
`endif
`ifdef ATTENTION_QUERY_VALUE_DISPATCH
        bit use_pair_values;
`endif
        int index, kvhead;
        for (int head=first_row;head<end_row;head++) begin
            kvhead=head/2;
            for (int query=0;query<batch_count;query+=2) begin
`ifdef ATTENTION_QUERY_VALUE_DISPATCH
                // Causal window shared by this actual pair, including its first query.
                use_pair_values=query_value_min_context>=0 && query+1<batch_count &&
                    attention_position+query+1>=query_value_min_context;
                if(use_pair_values) query_value_paired_count++;
                else query_value_fallback_count++;
`endif
                maximum[0]=-1.0e300;maximum[1]=-1.0e300;
                for (int t=0;t<=attention_position+query;t++) begin
                    sums[0]=0.0;sums[1]=0.0;
                    index=(attention_layer*4096+t)*1024+kvhead*128;
                    for (int j=0;j<128;j++) begin
                        key=attention_key_value(index+j);
                        sums[0]+=wide_activation[((head*128+j)*8+query)&32767]*key;
                        sums[1]+=wide_activation[((head*128+j)*8+query+1)&32767]*key;
                    end
                    pair_scores[t][0]=sums[0]/$sqrt(128.0);
                    pair_scores[t][1]=sums[1]/$sqrt(128.0);
                    if (pair_scores[t][0]>maximum[0]) maximum[0]=pair_scores[t][0];
                    if (pair_scores[t][1]>maximum[1]) maximum[1]=pair_scores[t][1];
                end
                // The second query alone can see its own newly published KV.
                if (query+1<batch_count) begin
                    sums[1]=0.0;
                    index=(attention_layer*4096+attention_position+query+1)*1024+kvhead*128;
                    for (int j=0;j<128;j++)
                        sums[1]+=wide_activation[((head*128+j)*8+query+1)&32767]*attention_key_value(index+j);
                    pair_scores[attention_position+query+1][1]=sums[1]/$sqrt(128.0);
                    if (pair_scores[attention_position+query+1][1]>maximum[1]) maximum[1]=pair_scores[attention_position+query+1][1];
                end
`ifdef ATTENTION_QUERY_VALUE_PAIRS
                denoms[0]=1.0;denoms[1]=1.0;
`endif
                for (int lane=0;lane<2 && query+lane<batch_count;lane++) begin
                    denom=0.0;
                    for (int t=0;t<=attention_position+query+lane;t++) begin
                        pair_scores[t][lane]=$exp(pair_scores[t][lane]-maximum[lane]);
                        denom+=pair_scores[t][lane];
                    end
`ifdef ATTENTION_QUERY_VALUE_PAIRS
                    denoms[lane]=denom;
`ifdef ATTENTION_QUERY_VALUE_DISPATCH
                    if(!use_pair_values) begin
                        for (int j=0;j<128;j++) result[head*128+j]=0.0;
                        for (int t=0;t<=attention_position+query+lane;t++) begin
                            scale=pair_scores[t][lane]/denom;
                            index=(attention_layer*4096+t)*1024+kvhead*128;
                            for (int j=0;j<128;j++) result[head*128+j]+=scale*attention_value_value(index+j);
                        end
                        for (int j=0;j<128;j++) wide_result[(head*128+j)*8+query+lane]=result[head*128+j];
                    end
`endif
`else
                    for (int j=0;j<128;j++) result[head*128+j]=0.0;
                    for (int t=0;t<=attention_position+query+lane;t++) begin
                        scale=pair_scores[t][lane]/denom;
                        index=(attention_layer*4096+t)*1024+kvhead*128;
                        for (int j=0;j<128;j++) result[head*128+j]+=scale*attention_value_value(index+j);
                    end
                    for (int j=0;j<128;j++) wide_result[(head*128+j)*8+query+lane]=result[head*128+j];
`endif
                end
`ifdef ATTENTION_QUERY_VALUE_PAIRS
`ifdef ATTENTION_QUERY_VALUE_DISPATCH
                if(use_pair_values) begin
`endif
                // Each query keeps its original ascending-t accumulation order.
                // Only independent queries share V loads and adjacent storage.
                for (int j=0;j<128;j++) begin
                    query_pair_values[j][0]=0.0;query_pair_values[j][1]=0.0;
                end
                for (int t=0;t<=attention_position+query;t++) begin
                    scales[0]=pair_scores[t][0]/denoms[0];
                    scales[1]=query+1<batch_count ? pair_scores[t][1]/denoms[1] : 0.0;
                    index=(attention_layer*4096+t)*1024+kvhead*128;
                    for (int j=0;j<128;j++) begin
                        value=attention_value_value(index+j);
                        query_pair_values[j][0]+=scales[0]*value;
                        query_pair_values[j][1]+=scales[1]*value;
                    end
                end
                if (query+1<batch_count) begin
                    scale=pair_scores[attention_position+query+1][1]/denoms[1];
                    index=(attention_layer*4096+attention_position+query+1)*1024+kvhead*128;
                    for (int j=0;j<128;j++)
                        query_pair_values[j][1]+=scale*attention_value_value(index+j);
                end
                for (int lane=0;lane<2 && query+lane<batch_count;lane++)
                    for (int j=0;j<128;j++)
                        wide_result[(head*128+j)*8+query+lane]=query_pair_values[j][lane];
`ifdef ATTENTION_QUERY_VALUE_DISPATCH
                end
`endif
`endif
            end
        end
    endtask
`endif
    task automatic attend_block_heads;
        real sum, maximum, denom, scale;
        int index, kvhead;
`ifdef ATTENTION_DOT_LANES8
        real dot_partial[8];
`endif
        for (int head=first_row;head<end_row;head++) begin
          for (int query=0;query<batch_count;query++) begin
            kvhead=head/2;
            maximum=-1.0e300;
            for (int t=0;t<=attention_position+query;t++) begin
                sum=0.0;
                index=(attention_layer*4096+t)*1024+kvhead*128;
`ifdef ATTENTION_DOT_LANES8
                for (int lane=0;lane<8;lane++) dot_partial[lane]=0.0;
                for (int unsigned j=0;j<128;j+=8) begin
                    dot_partial[0]+=wide_activation[((head*128+j+0)*8+query)&32767]*attention_key_value(index+j+0);
                    dot_partial[1]+=wide_activation[((head*128+j+1)*8+query)&32767]*attention_key_value(index+j+1);
                    dot_partial[2]+=wide_activation[((head*128+j+2)*8+query)&32767]*attention_key_value(index+j+2);
                    dot_partial[3]+=wide_activation[((head*128+j+3)*8+query)&32767]*attention_key_value(index+j+3);
                    dot_partial[4]+=wide_activation[((head*128+j+4)*8+query)&32767]*attention_key_value(index+j+4);
                    dot_partial[5]+=wide_activation[((head*128+j+5)*8+query)&32767]*attention_key_value(index+j+5);
                    dot_partial[6]+=wide_activation[((head*128+j+6)*8+query)&32767]*attention_key_value(index+j+6);
                    dot_partial[7]+=wide_activation[((head*128+j+7)*8+query)&32767]*attention_key_value(index+j+7);
                end
                for (int lane=0;lane<8;lane++) sum+=dot_partial[lane];
`else
                for (int j=0;j<128;j++)
                    sum+=wide_activation[((head*128+j)*8+query)&32767]*attention_key_value(index+j);
`endif
                attention_scores[t]=sum/$sqrt(128.0);
                if (attention_scores[t]>maximum) maximum=attention_scores[t];
            end
            denom=0.0;
            for (int t=0;t<=attention_position+query;t++) begin
                attention_scores[t]=$exp(attention_scores[t]-maximum);
                denom+=attention_scores[t];
            end
            for (int j=0;j<128;j++) result[head*128+j]=0.0;
            for (int t=0;t<=attention_position+query;t++) begin
                scale=attention_scores[t]/denom;
                index=(attention_layer*4096+t)*1024+kvhead*128;
                for (int j=0;j<128;j++)
                    result[head*128+j]+=scale*attention_value_value(index+j);
            end
            for (int j=0;j<128;j++) wide_result[(head*128+j)*8+query]=result[head*128+j];
          end
        end
    endtask
`endif
`endif
// Host validation guarantees nonnegative rows and columns. Unsigned loop
// variables keep index widening free of signed-extension bookkeeping.
`ifdef UNSIGNED_PROJECTION_INDEX
`define PROJECTION_INDEX int unsigned
`else
`define PROJECTION_INDEX int
`endif
`ifdef BATCH_PREFILL
    task automatic project_block;
        real partial[8][4];
        real sums[4];
        real weight;
        logic [2:0] lane;
        for (`PROJECTION_INDEX row=first_row;row<end_row;row++) begin
            for (int t=0;t<4;t++) begin
                sums[t]=0.0;
                for (int l=0;l<8;l++) partial[l][t]=0.0;
            end
`ifdef BLOCK_UNROLL
            if (reduction_lanes==8 && width%8==0) begin
                for (`PROJECTION_INDEX column=0;column<width && column<3065;column+=8) begin
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+0));
                    for (int t=0;t<4;t++)
                        partial[0][t]+=batch_activation[((column+0)*4+t)&16383]*weight;
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+1));
                    for (int t=0;t<4;t++)
                        partial[1][t]+=batch_activation[((column+1)*4+t)&16383]*weight;
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+2));
                    for (int t=0;t<4;t++)
                        partial[2][t]+=batch_activation[((column+2)*4+t)&16383]*weight;
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+3));
                    for (int t=0;t<4;t++)
                        partial[3][t]+=batch_activation[((column+3)*4+t)&16383]*weight;
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+4));
                    for (int t=0;t<4;t++)
                        partial[4][t]+=batch_activation[((column+4)*4+t)&16383]*weight;
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+5));
                    for (int t=0;t<4;t++)
                        partial[5][t]+=batch_activation[((column+5)*4+t)&16383]*weight;
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+6));
                    for (int t=0;t<4;t++)
                        partial[6][t]+=batch_activation[((column+6)*4+t)&16383]*weight;
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+7));
                    for (int t=0;t<4;t++)
                        partial[7][t]+=batch_activation[((column+7)*4+t)&16383]*weight;
                end
            end else begin
`endif
            for (`PROJECTION_INDEX column=0;column<width && column<3072;column++) begin
                weight=projection_weight_value(base+longint'(row)*width+longint'(column));
                lane=column[2:0];
                for (int t=0;t<batch_count;t++) begin
                    if (reduction_lanes==8 && width%8==0)
                        partial[lane][t]+=batch_activation[column*4+t]*weight;
                    else sums[t]+=batch_activation[column*4+t]*weight;
                end
            end
`ifdef BLOCK_UNROLL
            end
`endif
            for (int t=0;t<batch_count;t++) begin
                if (reduction_lanes==8 && width%8==0)
                    for (int l=0;l<8;l++) sums[t]+=partial[l][t];
                batch_result[row*4+t]=sums[t];
            end
        end
    endtask
`endif
`ifdef WIDE_BATCH
    task automatic project_wide_probe;
`ifdef WIDE_FOUR_LANES
        real partial4[4][`WIDE_BATCH], totals[`WIDE_BATCH], weight;
        for (`PROJECTION_INDEX row=first_row;row<end_row;row++) begin
            for (int t=0;t<`WIDE_BATCH;t++) begin
                totals[t]=0.0;
                for (int lane=0;lane<4;lane++) partial4[lane][t]=0.0;
            end
            if (reduction_lanes==8 && width%8==0) begin
                for (`PROJECTION_INDEX column=0;column<width && column<3069;column+=4) begin
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+0));
                    for (int t=0;t<`WIDE_BATCH;t++)
                        partial4[0][t]+=wide_activation[((column+0)*`WIDE_BATCH+t)&(4096*`WIDE_BATCH-1)]*weight;
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+1));
                    for (int t=0;t<`WIDE_BATCH;t++)
                        partial4[1][t]+=wide_activation[((column+1)*`WIDE_BATCH+t)&(4096*`WIDE_BATCH-1)]*weight;
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+2));
                    for (int t=0;t<`WIDE_BATCH;t++)
                        partial4[2][t]+=wide_activation[((column+2)*`WIDE_BATCH+t)&(4096*`WIDE_BATCH-1)]*weight;
                    weight=projection_weight_value(base+longint'(row)*width+(longint'(column)+3));
                    for (int t=0;t<`WIDE_BATCH;t++)
                        partial4[3][t]+=wide_activation[((column+3)*`WIDE_BATCH+t)&(4096*`WIDE_BATCH-1)]*weight;
                end
                for (int lane=0;lane<4;lane++) for (int t=0;t<`WIDE_BATCH;t++) totals[t]+=partial4[lane][t];
            end else begin
                for (`PROJECTION_INDEX column=0;column<width && column<3072;column++) begin
                    weight=projection_weight_value(base+longint'(row)*width+longint'(column));
                    for (int t=0;t<`WIDE_BATCH;t++)
                        totals[t]+=wide_activation[(column*`WIDE_BATCH+t)&(4096*`WIDE_BATCH-1)]*weight;
                end
            end
            for (int t=0;t<`WIDE_BATCH;t++) wide_result[row*`WIDE_BATCH+t]=totals[t];
        end
`else
        real lane_sums[`WIDE_BATCH], totals[`WIDE_BATCH], weight;
        for (`PROJECTION_INDEX row=first_row;row<end_row;row++) begin
            for (int t=0;t<`WIDE_BATCH;t++) totals[t]=0.0;
            if (reduction_lanes==8 && width%8==0) begin
                // Finish one lane at a time: preserve each lane's column order
                // and the original lane0..7 final reduction, with O(B) live sums.
                for (int lane=0;lane<8;lane++) begin
                    for (int t=0;t<`WIDE_BATCH;t++) lane_sums[t]=0.0;
                    for (`PROJECTION_INDEX column=lane;column<width && column<3072;column+=8) begin
                        weight=projection_weight_value(base+longint'(row)*width+longint'(column));
                        for (int t=0;t<`WIDE_BATCH;t++)
                            lane_sums[t]+=wide_activation[(column*`WIDE_BATCH+t)&(4096*`WIDE_BATCH-1)]*weight;
                    end
                    for (int t=0;t<`WIDE_BATCH;t++) totals[t]+=lane_sums[t];
                end
            end else begin
                for (`PROJECTION_INDEX column=0;column<width && column<3072;column++) begin
                    weight=projection_weight_value(base+longint'(row)*width+longint'(column));
                    for (int t=0;t<`WIDE_BATCH;t++)
                        totals[t]+=wide_activation[(column*`WIDE_BATCH+t)&(4096*`WIDE_BATCH-1)]*weight;
                end
            end
            for (int t=0;t<`WIDE_BATCH;t++) wide_result[row*`WIDE_BATCH+t]=totals[t];
        end
`endif
    endtask
`endif
    always @(posedge clk) begin
        real sum;
        real partial[8];
        if (start) begin
`ifdef PARALLEL_ATTENTION
`ifdef BATCH_PREFILL
`ifdef WIDE_BATCH
            if (operation==3) project_wide_probe();
            else
`endif
            if (operation==2) project_block();
            else
`endif
`ifdef BATCH_ATTENTION
            if (operation==5) begin
`ifdef ATTENTION_QK2
                attend_block_heads_qk2();
`else
                attend_block_heads();
`endif
            end
            else
`endif
`ifdef ATTENTION_HEAD_PAIRS
            if (operation==6) attend_head_pairs();
            else
`endif
            if (operation==1) attend_heads();
            else begin
`endif
            for (`PROJECTION_INDEX row=first_row;row<end_row;row++) begin
`ifdef BULK_ROW_WEIGHTS
                weight_row(base+longint'(row)*width,width,row_weights);
`endif
                if (reduction_lanes==8 && width%8==0) begin
                    for (int lane=0;lane<8;lane++) partial[lane]=0.0;
                    for (`PROJECTION_INDEX column=0;column<width && column<3065;column+=8) begin
`ifdef BULK_ROW_WEIGHTS
                        partial[0]+=`ACTIVATION_AT(column+0)*row_weights[column+0];
`else
                        partial[0]+=`ACTIVATION_AT(column+0)*projection_weight_value(base+longint'(row)*width+(longint'(column)+0));
`endif
`ifdef BULK_ROW_WEIGHTS
                        partial[1]+=`ACTIVATION_AT(column+1)*row_weights[column+1];
`else
                        partial[1]+=`ACTIVATION_AT(column+1)*projection_weight_value(base+longint'(row)*width+(longint'(column)+1));
`endif
`ifdef BULK_ROW_WEIGHTS
                        partial[2]+=`ACTIVATION_AT(column+2)*row_weights[column+2];
`else
                        partial[2]+=`ACTIVATION_AT(column+2)*projection_weight_value(base+longint'(row)*width+(longint'(column)+2));
`endif
`ifdef BULK_ROW_WEIGHTS
                        partial[3]+=`ACTIVATION_AT(column+3)*row_weights[column+3];
`else
                        partial[3]+=`ACTIVATION_AT(column+3)*projection_weight_value(base+longint'(row)*width+(longint'(column)+3));
`endif
`ifdef BULK_ROW_WEIGHTS
                        partial[4]+=`ACTIVATION_AT(column+4)*row_weights[column+4];
`else
                        partial[4]+=`ACTIVATION_AT(column+4)*projection_weight_value(base+longint'(row)*width+(longint'(column)+4));
`endif
`ifdef BULK_ROW_WEIGHTS
                        partial[5]+=`ACTIVATION_AT(column+5)*row_weights[column+5];
`else
                        partial[5]+=`ACTIVATION_AT(column+5)*projection_weight_value(base+longint'(row)*width+(longint'(column)+5));
`endif
`ifdef BULK_ROW_WEIGHTS
                        partial[6]+=`ACTIVATION_AT(column+6)*row_weights[column+6];
`else
                        partial[6]+=`ACTIVATION_AT(column+6)*projection_weight_value(base+longint'(row)*width+(longint'(column)+6));
`endif
`ifdef BULK_ROW_WEIGHTS
                        partial[7]+=`ACTIVATION_AT(column+7)*row_weights[column+7];
`else
                        partial[7]+=`ACTIVATION_AT(column+7)*projection_weight_value(base+longint'(row)*width+(longint'(column)+7));
`endif

                    end
                    sum=0.0;
                    for (int lane=0;lane<8;lane++) sum+=partial[lane];
                end else begin
                    sum=0.0;
                    for (`PROJECTION_INDEX column=0;column<width && column<3072;column++) begin
`ifdef BULK_ROW_WEIGHTS
                        sum+=`ACTIVATION_AT(column)*row_weights[column];
`else
                        sum+=`ACTIVATION_AT(column)*projection_weight_value(base+longint'(row)*width+longint'(column));
`endif
                    end
                end
                result[row]=sum;
            end
`ifdef PARALLEL_ATTENTION
            end
`endif
        end
    end
`undef PROJECTION_INDEX
`undef ACTIVATION_AT
endmodule
