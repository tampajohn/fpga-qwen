// Full Qwen3-0.6B functional SystemVerilog model. NOT synthesizable RTL.
// real arithmetic and transcendental functions establish a correctness baseline.
// A clock edge executes one token; that edge is NOT a hardware latency estimate.
// DPI supplies read-only checkpoint bytes and tensor offsets, never model math.
module qwen_behavioral (
    input logic clk, rst_n, start, do_logits, truncate_cache,
    input int truncate_to,
    input int token_id, parallel_workers,
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
    real projection_input[3072], projection_output[151936];
    localparam int H=1024, D=128, Q=2048, I=3072, L=28, C=4096, V=151936;
    real x[H], n[H], q[Q], k[H], v[H], a[Q], gate[I], up[I];
    real keys[], values[];
    real scores[C], cosine[64], sine[64];
    // Debug outputs are scalar transport only; all arithmetic is performed here.
    real logits[V];
    real last_hidden[H];
    int token_history[C];
    `include "sampling.svh"

    initial begin
        keys = new[L*C*H];
        values = new[L*C*H];
    end

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

    task automatic attention(input int layer);
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
            for (int j=0;j<D;j++) begin
                sum=0.0;
                for (int t=0;t<=position;t++)
                    sum+=scores[t]/denom*values[(layer*C+t)*H+kvhead*D+j];
                a[head*D+j]=sum;
            end
        end
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

    always @(posedge clk) begin : token_step
        real angle, sum;
        longint base;
        done=0;
        if (!rst_n) begin
            position=0; best_id=0; best_logit=0.0;
        end else if (truncate_cache) begin
            if (truncate_to<0 || truncate_to>position) $fatal(1,"Invalid KV truncation");
            position=truncate_to;
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
                attention(layer);
                normalize_hidden(3+layer*11+7);
                mlp(layer);
            end
            if (do_logits) begin
                normalize_hidden(1);
                for (int j=0;j<H;j++) last_hidden[j]=n[j];
                base=tensor_base(2);
                best_logit=-1.0e300; best_id=0;
                if (parallel_workers>0) begin
                    for (int j=0;j<H;j++) projection_input[j]=n[j];
                    dispatch_projection(base,V,H,projection_input,projection_output);
                end
                for (int r=0;r<V;r++) begin
                    sum=0.0;
                    if (parallel_workers>0) sum=projection_output[r];
                    else for (int j=0;j<H;j++) sum+=n[j]*weight_value(base+longint'(r)*H+longint'(j));
                    logits[r]=sum;
                    if (sum>best_logit) begin best_logit=sum; best_id=r; end
                end
                if (sample_enabled) begin
                    best_id=sample_chat_token(logits,token_history,position+1,sample_uniform,presence_penalty);
                    best_logit=logits[best_id];
                end
            end
            position+=1;
            done=1;
        end
    end
endmodule
