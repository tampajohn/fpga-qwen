// One coordinator-owned storage module; real ports keep FLOAT internal.
module fp32_kv_store #(
    parameter integer WORDS=28*4096*1024
)(input logic clk, write_enable,
  input integer write_base, read_index,
  input real write_keys[1024], write_values[1024],
  output real read_key, read_value);
    shortreal keys[WORDS], values[WORDS];
    always @(posedge clk) begin
        if(write_enable) begin
            if(write_base<0 || write_base>WORDS-1024) $fatal(1,"KV write span invalid");
            for(integer j=0;j<1024;j++) begin
                keys[write_base+j]=shortreal'(write_keys[j]);
                values[write_base+j]=shortreal'(write_values[j]);
            end
        end
    end
    always_comb begin
        read_key=0.0;read_value=0.0;
        if(read_index>=0 && read_index<WORDS) begin
            read_key=real'(keys[read_index]);read_value=real'(values[read_index]);
        end
    end
endmodule
