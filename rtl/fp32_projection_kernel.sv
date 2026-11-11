// Isolated projection experiment. All conversions and arithmetic remain in SV.
// Real ports preserve the existing transport ABI; only internal math is FP32.
module fp32_projection_kernel(
    input logic clk, start,
    input longint base,
    input int first_row, end_row, width,
    input real activation[4096*`KERNEL_BATCH],
    output real result[3072*`KERNEL_BATCH]
);
    import "DPI-C" function real weight_value_unchecked(input longint index);
    localparam int B=`KERNEL_BATCH;
    localparam int LANES=(B==8)?4:8;
    shortreal converted[4096*B];
    shortreal partial[8][B];
    shortreal totals[B];
    shortreal weight;
`define STEP(lane) \
    weight=shortreal'(weight_value_unchecked(base+longint'(row)*width+longint'(column+lane))); \
    for(int t=0;t<B;t++) \
        partial[lane][t]=shortreal'(partial[lane][t]+shortreal'(converted[((column+lane)*B+t)&(4096*B-1)]*weight));
    always @(posedge clk) begin
        if(start) begin
            for(int unsigned column=0;column<width;column++)
                for(int t=0;t<B;t++) converted[column*B+t]=shortreal'(activation[column*B+t]);
            for(int unsigned row=first_row;row<end_row;row++) begin
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
endmodule
