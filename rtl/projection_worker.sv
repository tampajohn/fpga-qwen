// Behavioral row partition. Arithmetic remains SystemVerilog, not host BLAS.
// Scalar and bulk-row transports are selectable; DPI never performs model arithmetic.
module projection_worker (
    input logic clk, start,
    input longint base,
`ifdef PARALLEL_ATTENTION
    input int operation, attention_layer, attention_position,
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
    task automatic attend_heads;
        real sum, maximum, denom, scale;
        int index, kvhead;
        for (int head=first_row;head<end_row;head++) begin
            kvhead=head/2;
            maximum=-1.0e300;
            for (int t=0;t<=attention_position;t++) begin
                sum=0.0;
                index=(attention_layer*4096+t)*1024+kvhead*128;
                for (int j=0;j<128;j++)
                    sum+=`ACTIVATION_AT(head*128+j)*attention_key_value(index+j);
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
