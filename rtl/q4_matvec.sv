// Experimental W4A8 matrix engine. Eight token lanes share each weight beat.
// Arithmetic: signed int4 weights * signed int8 activations -> signed int32.
// Row quantization scales are applied outside this integer primitive.
module q4_matvec #(
    parameter int MAX_K = 3072,
    parameter int TOKENS = 8,
    parameter int LANES = 16
) (
    input logic clk, rst_n,
    input logic x_we,
    input logic [11:0] x_addr,
    input logic [TOKENS*8-1:0] x_data,
    input logic start,
    input logic [3:0] token_count,
    input logic [LANES*4-1:0] w_data,
    input logic w_valid, w_last,
    output logic w_ready,
    output logic [TOKENS*32-1:0] y_data,
    output logic y_valid,
    input logic y_ready
);
    typedef enum logic [1:0] {IDLE, RUN, RESULT} state_t;
    state_t state;
    logic signed [7:0] activations [TOKENS][MAX_K];
    logic signed [31:0] accum [TOKENS];
    logic signed [31:0] partial [TOKENS];
    logic [3:0] active_tokens;
    int unsigned offset;

    assign w_ready = state == RUN;
    assign y_valid = state == RESULT;
    for (genvar t = 0; t < TOKENS; t++) begin : outputs
        assign y_data[t*32 +: 32] = accum[t];
    end

    always_comb begin
        for (int t = 0; t < TOKENS; t++) begin
            partial[t] = 0;
            if (state == RUN && t < int'(active_tokens)) begin
                for (int l = 0; l < LANES; l++) begin
                    if (offset + l < MAX_K)
                        partial[t] += 32'($signed(w_data[l*4 +: 4])) *
                                      32'(activations[t][offset+l]);
                end
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            active_tokens <= 0;
            offset <= 0;
            for (int t = 0; t < TOKENS; t++) accum[t] <= 0;
        end else begin
            if (x_we && state == IDLE && int'(x_addr) < MAX_K)
                for (int t = 0; t < TOKENS; t++)
                    activations[t][x_addr] <= $signed(x_data[t*8 +: 8]);
            case (state)
                IDLE: if (start) begin
                    state <= RUN;
                    active_tokens <= token_count;
                    offset <= 0;
                    for (int t = 0; t < TOKENS; t++) accum[t] <= 0;
                end
                RUN: if (w_valid && w_ready) begin
                    for (int t = 0; t < TOKENS; t++)
                        accum[t] <= accum[t] + partial[t];
                    if (w_last) state <= RESULT;
                    else offset <= offset + LANES;
                end
                RESULT: if (y_ready) state <= IDLE;
                default: state <= IDLE;
            endcase
        end
    end
endmodule
