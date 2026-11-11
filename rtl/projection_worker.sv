// Behavioral row partition. Arithmetic remains SystemVerilog, not host BLAS.
// Each worker owns independent result rows; every dot product retains its order.
module projection_worker (
    input logic clk, start,
    input longint base,
    input int first_row, end_row, width,
    input real activation[3072],
    output real result[151936]
);
    import "DPI-C" function real weight_value(input longint index);
    always @(posedge clk) begin
        real sum;
        if (start) begin
            for (int row=first_row;row<end_row;row++) begin
                sum=0.0;
                for (int column=0;column<width;column++)
                    sum+=activation[column]*weight_value(base+longint'(row)*width+longint'(column));
                result[row]=sum;
            end
        end
    end
endmodule
