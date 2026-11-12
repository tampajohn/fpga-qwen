// Isolated projection experiment. All conversions and arithmetic remain in SV.
// Real ports preserve the existing transport ABI; only internal math is FP32.
`ifndef KERNEL_ROWS
`define KERNEL_ROWS 3072
`endif
module fp32_projection_kernel(
    input logic clk, start,
    input longint base,
    input int first_row, end_row, width,
    input real activation[4096*`KERNEL_BATCH],
    output real result[`KERNEL_ROWS*`KERNEL_BATCH]
);
    import "DPI-C" function real weight_value_unchecked(input longint index);
    localparam int B=`KERNEL_BATCH;
    localparam int LANES=(B==8)?4:8;
    shortreal converted[4096*B];
    shortreal partial[8][B];
    shortreal totals[B];
    shortreal weight;
`ifdef FP32_EXPLICIT_FMA
// Nonportable experimental primitive: one binary32 rounding per update.
// Independent token layout, column lanes and final lane sum stay unchanged.
`define ACCUM(lane, t) \
    partial[lane][t]=$verilator_fma32(converted[((column+lane)*B+t)&(4096*B-1)],weight,partial[lane][t]);
`else
`define ACCUM(lane, t) \
    partial[lane][t]=shortreal'(partial[lane][t]+shortreal'(converted[((column+lane)*B+t)&(4096*B-1)]*weight));
`endif
`define STEP(lane) \
    weight=shortreal'(weight_value_unchecked(base+longint'(row)*width+longint'(column+lane))); \
    for(int t=0;t<B;t++) `ACCUM(lane,t)
`ifdef FP32_ROW_PAIRS
    // Two independent output rows reuse each converted activation. Every row
    // retains eight column lanes and separately rounded multiplication/addition.
    // No packing, host arithmetic, horizontal reduction or cross-row sum.
    shortreal row_partial[2][8];
    shortreal row_total[2];
    initial begin
        if(B!=1) $fatal(1,"FP32 row pairs require scalar B1");
`ifdef FP32_EXPLICIT_FMA
        $fatal(1,"FP32 row pairs preserve separate multiplication/addition");
`endif
    end
`define ROW_PAIR_STEP(lane) \
    x=converted[(column+lane)&4095]; \
    w0=shortreal'(weight_value_unchecked(base+longint'(first)*width+longint'(column+lane))); \
    w1=shortreal'(weight_value_unchecked(base+(longint'(first)+1)*width+longint'(column+lane))); \
    row_partial[0][lane]=shortreal'(row_partial[0][lane]+shortreal'(x*w0)); \
    row_partial[1][lane]=shortreal'(row_partial[1][lane]+shortreal'(x*w1));
    task automatic project_row_pair(input int unsigned first);
        shortreal x,w0,w1;
        for(int r=0;r<2;r++) begin
            row_total[r]=shortreal'(0.0);
            for(int lane=0;lane<8;lane++) row_partial[r][lane]=shortreal'(0.0);
        end
        for(int unsigned column=0;column<width;column+=8) begin
            `ROW_PAIR_STEP(0)
            `ROW_PAIR_STEP(1)
            `ROW_PAIR_STEP(2)
            `ROW_PAIR_STEP(3)
            `ROW_PAIR_STEP(4)
            `ROW_PAIR_STEP(5)
            `ROW_PAIR_STEP(6)
            `ROW_PAIR_STEP(7)
        end
        for(int r=0;r<2;r++) begin
            for(int lane=0;lane<8;lane++)
                row_total[r]=shortreal'(row_total[r]+row_partial[r][lane]);
            result[first+r]=real'(row_total[r]);
        end
    endtask
`undef ROW_PAIR_STEP
`endif
    always @(posedge clk) begin
        if(start) begin
            for(int unsigned column=0;column<width;column++)
                for(int t=0;t<B;t++) converted[column*B+t]=shortreal'(activation[column*B+t]);
`ifdef FP32_ROW_PAIRS
            // Each worker owns [first_row,end_row); never pair across its boundary.
            for(int unsigned row=first_row;row+1<end_row;row+=2)
                project_row_pair(row);
            // The existing scalar kernel handles a partition's odd final row.
            for(int unsigned row=first_row+((end_row-first_row)/2)*2;row<end_row;row++) begin
`else
            for(int unsigned row=first_row;row<end_row;row++) begin
`endif
                for(int t=0;t<B;t++) begin
                    totals[t]=shortreal'(0.0);
                    for(int lane=0;lane<8;lane++)partial[lane][t]=shortreal'(0.0);
                end
                for(int unsigned column=0;column<width;column+=LANES) begin
                    `STEP(0)
                    `STEP(1)
                    `STEP(2)
                    `STEP(3)
                    if(LANES==8)begin
                        `STEP(4)
                        `STEP(5)
                        `STEP(6)
                        `STEP(7)
                    end
                end
                for(int lane=0;lane<LANES;lane++)
                    for(int t=0;t<B;t++)totals[t]=shortreal'(totals[t]+partial[lane][t]);
                for(int t=0;t<B;t++)result[row*B+t]=real'(totals[t]);
            end
        end
    end
`undef STEP
`undef ACCUM
endmodule
