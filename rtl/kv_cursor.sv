// One sequence's speculative KV visibility/ownership metadata. Payload storage
// is external. write completion must cover every KV value for the marked tokens.
// Commands: 0 reserve, 1 mark written, 2 commit prefix, 3 abort, 4 truncate prefix.
module kv_cursor #(
    parameter int CAPACITY = 4096,
    parameter int MAX_DRAFT = 8
) (
    input logic clk, rst_n,
    input logic cmd_valid,
    output logic cmd_ready,
    input logic [2:0] cmd_op,
    input logic [15:0] cmd_count,
    output logic rsp_valid,
    input logic rsp_ready,
    output logic rsp_error,
    output logic active,
    output logic [15:0] committed,
    output logic [15:0] reserved,
    output logic [15:0] written,
    output logic [15:0] visible_end
);
    assign cmd_ready = !rsp_valid;
    assign visible_end = committed + written;
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rsp_valid <= 0;
            rsp_error <= 0;
            active <= 0;
            committed <= 0;
            reserved <= 0;
            written <= 0;
        end else begin
            if (rsp_valid && rsp_ready) rsp_valid <= 0;
            if (cmd_valid && cmd_ready) begin
                rsp_valid <= 1;
                rsp_error <= 0;
                case (cmd_op)
                    0: if (!active && cmd_count > 0 && int'(cmd_count) <= MAX_DRAFT &&
                           int'(committed) + int'(cmd_count) <= CAPACITY) begin
                        active <= 1;
                        reserved <= cmd_count;
                        written <= 0;
                    end else rsp_error <= 1;
                    1: if (active && cmd_count > 0 &&
                           int'(written) + int'(cmd_count) <= int'(reserved)) begin
                        written <= written + cmd_count;
                    end else rsp_error <= 1;
                    2: if (active && written == reserved && cmd_count <= reserved) begin
                        committed <= committed + cmd_count;
                        active <= 0;
                        reserved <= 0;
                        written <= 0;
                    end else rsp_error <= 1;
                    3: if (active && cmd_count == 0) begin
                        active <= 0;
                        reserved <= 0;
                        written <= 0;
                    end else rsp_error <= 1;
                    4: if (!active && cmd_count <= committed) begin
                        committed <= cmd_count;
                    end else rsp_error <= 1;
                    default: rsp_error <= 1;
                endcase
            end
        end
    end
endmodule
