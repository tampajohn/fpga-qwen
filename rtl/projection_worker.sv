// Behavioral row partition. Arithmetic remains SystemVerilog, not host BLAS.
// Scalar and bulk-row transports are selectable; DPI never performs model arithmetic.
module projection_worker (
    input logic clk, start,
    input longint base,
`ifdef PARALLEL_ATTENTION
    input int operation, attention_layer, attention_position,
`endif
    input int first_row, end_row, width, reduction_lanes,
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
    always @(posedge clk) begin
        real sum;
        real partial[8];
        if (start) begin
`ifdef PARALLEL_ATTENTION
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
