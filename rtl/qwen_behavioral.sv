// Full Qwen3-0.6B functional SystemVerilog model. NOT synthesizable RTL.
// real arithmetic and transcendental functions establish a correctness baseline.
// A clock edge executes one token; that edge is NOT a hardware latency estimate.
// DPI supplies read-only checkpoint bytes and tensor offsets, never model math.
module qwen_behavioral (
    input logic clk, rst_n, start, do_logits, truncate_cache,
    input int truncate_to,
    input int token_id, parallel_workers,
`ifdef BATCH_PREFILL
    input logic block_start,
    input int block_count,
    input int block_tokens[4],
    input real block_uniforms[4],
    output int block_ids[4], block_positions[4],
    output real block_logits[4],
`endif
    input logic sample_enabled,
    input real sample_uniform, presence_penalty,
    output logic done,
    output int best_id,
    output int position,
    output real best_logit
);
    import "DPI-C" function real weight_value(input longint index);
    import "DPI-C" function longint tensor_base(input int slot);
    import "DPI-C" function void dispatch_projection(input longint base, input int rows, width,
                                                      input real activation[], output real result[]);
`ifdef PARALLEL_ATTENTION
    import "DPI-C" function void dispatch_attention(input int layer, position,
        input real query[], output real attention_result[]);
    real attention_output[2048];
`endif
`ifdef BATCH_PREFILL
    import "DPI-C" function void dispatch_block_projection(input longint base,
        input int rows, width, count, input real activation[], output real result[]);
    real block_x[1024][4], block_n[1024][4], block_q[2048][4];
    real block_k[1024][4], block_v[1024][4], block_a[2048][4];
    real block_gate[3072][4], block_up[3072][4];
    real block_input[12288], block_output[12288];
`endif
    real projection_input[3072];
`ifdef COMPACT_PROJECTION_BUFFER
    // DPI output arrays are initialized/copied at their declared size. Hidden
    // projections have at most 3072 rows; only the final vocabulary needs V.
    real projection_output[3072], vocabulary_output[151936];
`else
    real projection_output[151936];
`endif
    localparam int H=1024, D=128, Q=2048, I=3072, L=28, C=4096, V=151936;
    real x[H], n[H], q[Q], k[H], v[H], a[Q], gate[I], up[I];
`ifdef STATIC_KV
    // Exact dimensions avoid deque-backed dynamic indexing without changing
    // attention arithmetic or rounding. Do not pad these giant arrays.
    real keys[L*C*H], values[L*C*H];
`else
    real keys[], values[];
`endif
    real scores[C], cosine[64], sine[64];
    // Debug outputs are scalar transport only; all arithmetic is performed here.
    real logits[V];
    real last_hidden[H];
    int token_history[C];
    `include "sampling.svh"

`ifndef STATIC_KV
    initial begin
        keys = new[L*C*H];
        values = new[L*C*H];
    end
`endif

    task automatic normalize_hidden(input int slot);
        real sum, scale;
        longint base;
        sum=0.0;
        for (int j=0;j<H;j++) sum += x[j]*x[j];
        scale=1.0/$sqrt(sum/H+1.0e-6);
        base=tensor_base(slot);
        for (int j=0;j<H;j++) n[j]=x[j]*scale*weight_value(base+longint'(j));
    endtask

    // Separate arrays keep the memory layout explicit and avoid host BLAS calls.
    task automatic project_qkv(input int layer);
        real sum;
        longint base;
        base=tensor_base(3+layer*11+1);
        if (parallel_workers>0) begin
            for (int j=0;j<H;j++) projection_input[j]=n[j];
            dispatch_projection(base,Q,H,projection_input,projection_output);
            for (int r=0;r<Q;r++) q[r]=projection_output[r];
        end else begin
        for (int r=0;r<Q;r++) begin
            sum=0.0;
            for (int j=0;j<H;j++) sum+=n[j]*weight_value(base+longint'(r)*H+longint'(j));
            q[r]=sum;
        end
        end
        base=tensor_base(3+layer*11+2);
        if (parallel_workers>0) begin
            for (int j=0;j<H;j++) projection_input[j]=n[j];
            dispatch_projection(base,H,H,projection_input,projection_output);
            for (int r=0;r<H;r++) k[r]=projection_output[r];
        end else begin
        for (int r=0;r<H;r++) begin
            sum=0.0;
            for (int j=0;j<H;j++) sum+=n[j]*weight_value(base+longint'(r)*H+longint'(j));
            k[r]=sum;
        end
        end
        base=tensor_base(3+layer*11+3);
        if (parallel_workers>0) begin
            for (int j=0;j<H;j++) projection_input[j]=n[j];
            dispatch_projection(base,H,H,projection_input,projection_output);
            for (int r=0;r<H;r++) v[r]=projection_output[r];
        end else begin
        for (int r=0;r<H;r++) begin
            sum=0.0;
            for (int j=0;j<H;j++) sum+=n[j]*weight_value(base+longint'(r)*H+longint'(j));
            v[r]=sum;
        end
        end
    endtask

    task automatic attention(input int layer, input bit project_result);
        real sum, scale, left_value, right_value, maximum, denom;
        longint base;
        int index, kvhead;
        base=tensor_base(3+layer*11+5);
        for (int head=0;head<16;head++) begin
            sum=0.0;
            for (int j=0;j<D;j++) sum+=q[head*D+j]*q[head*D+j];
            scale=1.0/$sqrt(sum/D+1.0e-6);
            for (int j=0;j<D;j++) q[head*D+j]*=scale*weight_value(base+longint'(j));
            for (int j=0;j<64;j++) begin
                left_value=q[head*D+j]; right_value=q[head*D+j+64];
                q[head*D+j]=left_value*cosine[j]-right_value*sine[j];
                q[head*D+j+64]=right_value*cosine[j]+left_value*sine[j];
            end
        end
        base=tensor_base(3+layer*11+6);
        for (int head=0;head<8;head++) begin
            sum=0.0;
            for (int j=0;j<D;j++) sum+=k[head*D+j]*k[head*D+j];
            scale=1.0/$sqrt(sum/D+1.0e-6);
            for (int j=0;j<D;j++) k[head*D+j]*=scale*weight_value(base+longint'(j));
            for (int j=0;j<64;j++) begin
                left_value=k[head*D+j]; right_value=k[head*D+j+64];
                k[head*D+j]=left_value*cosine[j]-right_value*sine[j];
                k[head*D+j+64]=right_value*cosine[j]+left_value*sine[j];
            end
        end
        index=(layer*C+position)*H;
        for (int j=0;j<H;j++) begin
            keys[index+j]=k[j]; values[index+j]=v[j];
        end
`ifdef PARALLEL_ATTENTION
        if (parallel_workers>0) begin
            dispatch_attention(layer,position,q,attention_output);
            for (int j=0;j<Q;j++) a[j]=attention_output[j];
        end else begin
`endif
        for (int head=0;head<16;head++) begin
            kvhead=head/2;
            maximum=-1.0e300;
            for (int t=0;t<=position;t++) begin
                sum=0.0;
                index=(layer*C+t)*H+kvhead*D;
                for (int j=0;j<D;j++) sum+=q[head*D+j]*keys[index+j];
                scores[t]=sum/$sqrt(128.0);
                if (scores[t]>maximum) maximum=scores[t];
            end
            denom=0.0;
            for (int t=0;t<=position;t++) begin
                scores[t]=$exp(scores[t]-maximum);
                denom+=scores[t];
            end
`ifdef FAST_ATTENTION
            // Walk each contiguous value vector once. Each output keeps the same
            // ascending-time reduction order as the reference implementation.
            for (int j=0;j<D;j++) a[head*D+j]=0.0;
            for (int t=0;t<=position;t++) begin
                scale=scores[t]/denom;
                index=(layer*C+t)*H+kvhead*D;
                for (int j=0;j<D;j++)
                    a[head*D+j]+=scale*values[index+j];
            end
`else
            for (int j=0;j<D;j++) begin
                sum=0.0;
                for (int t=0;t<=position;t++)
                    sum+=scores[t]/denom*values[(layer*C+t)*H+kvhead*D+j];
                a[head*D+j]=sum;
            end
`endif
        end
`ifdef PARALLEL_ATTENTION
        end
`endif
        if (project_result) begin
        base=tensor_base(3+layer*11+4);
        if (parallel_workers>0) begin
            for (int j=0;j<Q;j++) projection_input[j]=a[j];
            dispatch_projection(base,H,Q,projection_input,projection_output);
            for (int r=0;r<H;r++) x[r]+=projection_output[r];
        end else begin
        for (int r=0;r<H;r++) begin
            sum=0.0;
            for (int j=0;j<Q;j++) sum+=a[j]*weight_value(base+longint'(r)*Q+longint'(j));
            x[r]+=sum;
        end
        end
        end
    endtask

    task automatic mlp(input int layer);
        real sum, g, sigmoid;
        longint base;
        base=tensor_base(3+layer*11+8);
        if (parallel_workers>0) begin
            for (int j=0;j<H;j++) projection_input[j]=n[j];
            dispatch_projection(base,I,H,projection_input,projection_output);
        end
        for (int r=0;r<I;r++) begin
            sum=0.0;
            if (parallel_workers>0) sum=projection_output[r];
            else for (int j=0;j<H;j++) sum+=n[j]*weight_value(base+longint'(r)*H+longint'(j));
            // Stable SiLU avoids exp overflow for large negative inputs.
            if (sum>=0.0) sigmoid=1.0/(1.0+$exp(-sum));
            else begin g=$exp(sum); sigmoid=g/(1.0+g); end
            gate[r]=sum*sigmoid;
        end
        base=tensor_base(3+layer*11+9);
        if (parallel_workers>0) begin
            for (int j=0;j<H;j++) projection_input[j]=n[j];
            dispatch_projection(base,I,H,projection_input,projection_output);
        end
        for (int r=0;r<I;r++) begin
            sum=0.0;
            if (parallel_workers>0) sum=projection_output[r];
            else for (int j=0;j<H;j++) sum+=n[j]*weight_value(base+longint'(r)*H+longint'(j));
            up[r]=gate[r]*sum;
        end
        base=tensor_base(3+layer*11+10);
        if (parallel_workers>0) begin
            for (int j=0;j<I;j++) projection_input[j]=up[j];
            dispatch_projection(base,H,I,projection_input,projection_output);
            for (int r=0;r<H;r++) x[r]+=projection_output[r];
        end else begin
        for (int r=0;r<H;r++) begin
            sum=0.0;
            for (int j=0;j<I;j++) sum+=up[j]*weight_value(base+longint'(r)*I+longint'(j));
            x[r]+=sum;
        end
        end
    endtask

    task automatic finish_logits(input bit enabled, input real uniform_value);
        real sum;
        longint base;
                normalize_hidden(1);
                for (int j=0;j<H;j++) last_hidden[j]=n[j];
                base=tensor_base(2);
                best_logit=-1.0e300; best_id=0;
                if (parallel_workers>0) begin
                    for (int j=0;j<H;j++) projection_input[j]=n[j];
`ifdef COMPACT_PROJECTION_BUFFER
                    dispatch_projection(base,V,H,projection_input,vocabulary_output);
`else
                    dispatch_projection(base,V,H,projection_input,projection_output);
`endif
                end
                for (int r=0;r<V;r++) begin
                    sum=0.0;
`ifdef COMPACT_PROJECTION_BUFFER
                    if (parallel_workers>0) sum=vocabulary_output[r];
`else
                    if (parallel_workers>0) sum=projection_output[r];
`endif
                    else for (int j=0;j<H;j++) sum+=n[j]*weight_value(base+longint'(r)*H+longint'(j));
                    logits[r]=sum;
                    if (sum>best_logit) begin best_logit=sum; best_id=r; end
                end
                if (enabled) begin
                    best_id=sample_chat_token(logits,token_history,position+1,uniform_value,presence_penalty);
                    best_logit=logits[best_id];
                end
    endtask

`ifdef BATCH_PREFILL
    task automatic block_project(input int slot, rows, width);
        real sum;
        longint base;
        base=tensor_base(slot);
        if (parallel_workers>0)
            dispatch_block_projection(base,rows,width,block_count,block_input,block_output);
        else begin
            for (int r=0;r<rows;r++) begin
                for (int t=0;t<block_count;t++) begin
                    sum=0.0;
                    for (int j=0;j<width;j++)
                        sum+=block_input[j*4+t]*weight_value(base+longint'(r)*width+longint'(j));
                    block_output[r*4+t]=sum;
                end
            end
        end
    endtask

    task automatic block_normalize(input int slot);
        for (int t=0;t<block_count;t++) begin
            for (int j=0;j<H;j++) x[j]=block_x[j][t];
            normalize_hidden(slot);
            for (int j=0;j<H;j++) begin
                block_n[j][t]=n[j];block_input[j*4+t]=n[j];
            end
        end
    endtask

    task automatic run_block;
        int initial_position;
        real angle, sum, g, sigmoid;
        longint base;
        initial_position=position;
        if (block_count<1 || block_count>4 || position+block_count>C)
            $fatal(1,"Invalid block count or context full");
        for (int t=0;t<block_count;t++) begin
            if (block_tokens[t]<0 || block_tokens[t]>=V) $fatal(1,"Invalid block token");
            token_history[initial_position+t]=block_tokens[t];
            base=tensor_base(0)+longint'(block_tokens[t])*H;
            for (int j=0;j<H;j++) block_x[j][t]=weight_value(base+longint'(j));
        end
        // Each layer finishes all block hidden states before the next starts.
        for (int layer=0;layer<L;layer++) begin
            block_normalize(3+layer*11);
            block_project(3+layer*11+1,Q,H);
            for (int j=0;j<Q;j++) for (int t=0;t<block_count;t++) block_q[j][t]=block_output[j*4+t];
            block_project(3+layer*11+2,H,H);
            for (int j=0;j<H;j++) for (int t=0;t<block_count;t++) block_k[j][t]=block_output[j*4+t];
            block_project(3+layer*11+3,H,H);
            for (int j=0;j<H;j++) for (int t=0;t<block_count;t++) block_v[j][t]=block_output[j*4+t];
            for (int t=0;t<block_count;t++) begin
                position=initial_position+t;
                for (int j=0;j<Q;j++) q[j]=block_q[j][t];
                for (int j=0;j<H;j++) begin k[j]=block_k[j][t];v[j]=block_v[j][t];end
                for (int j=0;j<64;j++) begin
                    angle=real'(position)/(1000000.0**(real'(2*j)/128.0));
                    cosine[j]=$cos(angle);sine[j]=$sin(angle);
                end
                // Writes this token's KV, then reads only <=position.
                attention(layer,0);
                for (int j=0;j<Q;j++) block_a[j][t]=a[j];
            end
            for (int j=0;j<Q;j++) for (int t=0;t<block_count;t++) block_input[j*4+t]=block_a[j][t];
            block_project(3+layer*11+4,H,Q);
            for (int j=0;j<H;j++) for (int t=0;t<block_count;t++) block_x[j][t]+=block_output[j*4+t];
            block_normalize(3+layer*11+7);
            block_project(3+layer*11+8,I,H);
            for (int j=0;j<I;j++) begin
                for (int t=0;t<block_count;t++) begin
                    sum=block_output[j*4+t];
                    if (sum>=0.0) sigmoid=1.0/(1.0+$exp(-sum));
                    else begin g=$exp(sum);sigmoid=g/(1.0+g);end
                    block_gate[j][t]=sum*sigmoid;
                end
            end
            block_project(3+layer*11+9,I,H);
            for (int j=0;j<I;j++) for (int t=0;t<block_count;t++) begin
                block_up[j][t]=block_gate[j][t]*block_output[j*4+t];
                block_input[j*4+t]=block_up[j][t];
            end
            block_project(3+layer*11+10,H,I);
            for (int j=0;j<H;j++) for (int t=0;t<block_count;t++) block_x[j][t]+=block_output[j*4+t];
        end
        // Vocabulary projection stays single-position initially; no huge BxV ports.
        for (int t=0;t<block_count;t++) begin
            position=initial_position+t;
            if (do_logits) begin
                for (int j=0;j<H;j++) x[j]=block_x[j][t];
                finish_logits(sample_enabled,block_uniforms[t]);
                block_ids[t]=best_id;block_logits[t]=best_logit;
            end else begin block_ids[t]=0;block_logits[t]=0.0;end
            block_positions[t]=position+1;
        end
        position=initial_position+block_count;
        done=1;
    endtask
`endif

    always @(posedge clk) begin : token_step
        real angle, sum;
        longint base;
        done=0;
        if (!rst_n) begin
            position=0; best_id=0; best_logit=0.0;
        end else if (truncate_cache) begin
            if (truncate_to<0 || truncate_to>position) $fatal(1,"Invalid KV truncation");
            position=truncate_to;
`ifdef BATCH_PREFILL
        end else if (block_start) begin
            run_block();
`endif
        end else if (start) begin
            if (position>=C || token_id<0 || token_id>=V) $fatal(1,"Invalid token or context full");
            token_history[position]=token_id;
            base=tensor_base(0)+longint'(token_id)*H;
            for (int j=0;j<H;j++) x[j]=weight_value(base+longint'(j));
            for (int j=0;j<64;j++) begin
                angle=real'(position)/(1000000.0**(real'(2*j)/128.0));
                cosine[j]=$cos(angle); sine[j]=$sin(angle);
            end
            for (int layer=0;layer<L;layer++) begin
                normalize_hidden(3+layer*11);
                project_qkv(layer);
                attention(layer,1);
                normalize_hidden(3+layer*11+7);
                mlp(layer);
            end
            if (do_logits) finish_logits(sample_enabled,sample_uniform);
            position+=1;
            done=1;
        end
    end
endmodule
