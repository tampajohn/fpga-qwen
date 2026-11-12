// Experimental storage only. All narrowing/widening occurs in SystemVerilog.
module fp32_kv_fixture #(
    parameter integer WORDS = 16
)(
    input logic clk,
    input logic write_enable,
    input integer write_index,
    input real write_key,
    input real write_value,
    input integer read_index,
    output real read_key,
    output real read_value,
    output logic write_valid,
    output logic read_valid
);
    shortreal keys[WORDS];
    shortreal values[WORDS];
    initial begin
        if (WORDS <= 0) $fatal(1, "WORDS must be positive");
        for (integer i=0; i<WORDS; i++) begin
            keys[i]=shortreal'(0.0);
            values[i]=shortreal'(0.0);
        end
    end
    always @(posedge clk) begin
        write_valid=0;
        if (write_enable && write_index >= 0 && write_index < WORDS) begin
            keys[write_index]=shortreal'(write_key);
            values[write_index]=shortreal'(write_value);
            write_valid=1;
        end
    end
    always_comb begin
        read_valid=0;
        read_key=0.0;
        read_value=0.0;
        if (read_index >= 0 && read_index < WORDS) begin
            read_key=real'(keys[read_index]);
            read_value=real'(values[read_index]);
            read_valid=1;
        end
    end
endmodule
