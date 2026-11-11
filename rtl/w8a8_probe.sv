// Isolated behavioral W8A8 experiment. Quantization and all model math stay SV.
module w8a8_probe (
    input logic clk, start,
    input int operation, rows, width, tokens, fixture,
    input real weights[49152], embedding[4096], norm_weight[1024],
    output int failures,
    output real result[64], reference_result[64], float_result[64],
    output real max_error, relative_rmse
);
    localparam int R=16,K=3072,B=4,G=64,NG=48;
    // Power-of-two padding makes every 12-bit feature index bounded.
    byte signed wq[R][4096], xq[B][4096];
    real sw[R][NG], sx[B][NG], x[B][K];
    int signed group_acc[R][B][NG];
    real worker_activation[4096], worker_batch_activation[16384];
    real worker_result[151936], worker_batch_result[12288];
    for (genvar feature=0;feature<4096;feature++) begin : gen_feature
        if(feature<K) begin : gen_valid
            assign worker_activation[feature]=x[0][feature];
            for(genvar token_index=0;token_index<B;token_index++) begin : gen_token
                assign worker_batch_activation[feature*B+token_index]=x[token_index][feature];
            end
        end else begin : gen_padding
            assign worker_activation[feature]=0.0;
            for(genvar token_index=0;token_index<B;token_index++) begin : gen_token
                assign worker_batch_activation[feature*B+token_index]=0.0;
            end
        end
    end
    projection_worker deployed_worker(
        .clk(clk),.start(start && operation==9),.base(64'd0),
        .operation(tokens==1?0:2),.attention_layer(0),.attention_position(0),
        .first_row(0),.end_row(rows),.width(width),.reduction_lanes(8),
        .batch_count(tokens),.batch_activation(worker_batch_activation),
        .batch_result(worker_batch_result),.activation(worker_activation),.result(worker_result)
    );


    function automatic int signed quantize(input real value, scale);
        real z, magnitude, fraction;
        int signed integer_part;
        z=value/scale;
        if (z>=127.0) return 127;
        if (z<=-127.0) return -127;
        magnitude=z<0.0?-z:z;
        integer_part=int'($floor(magnitude));
        fraction=magnitude-real'(integer_part);
        if (fraction>0.5 || (fraction==0.5 && (integer_part&1)!=0)) integer_part+=1;
        return z<0.0?-integer_part:integer_part;
    endfunction

    // Independent signed-truncation formulation for the preparation oracle.
    function automatic longint signed quantize_reference(input real value, scale);
        real z, remainder;
        longint signed truncated;
        z=value/scale;
        if (z>=127.0) return 127;
        if (z<=-127.0) return -127;
        truncated=64'($rtoi(z));
        remainder=z-real'(truncated);
        if (remainder>0.5 || (remainder==0.5 && truncated%2!=0)) truncated+=1;
        if (remainder< -0.5 || (remainder== -0.5 && truncated%2!=0)) truncated-=1;
        return truncated;
    endfunction

    task automatic prepare_activations;
        real maximum, value;
        for (int t=0;t<tokens;t++) begin
            for (int g=0;g<(width+G-1)/G;g++) begin
                maximum=0.0;
                for (int j=0;j<G;j++) if (g*G+j<width) begin
                    value=x[t][g*G+j];
                    if ((value-value)!=0.0) $fatal(1,"Nonfinite activation");
                    if (value<0.0) value=-value;
                    if (value>maximum) maximum=value;
                end
                sx[t][g]=maximum==0.0?1.0:maximum/127.0;
                for (int j=0;j<G;j++) begin
                    if (g*G+j<width) xq[t][12'(g*G+j)]=8'(quantize(x[t][g*G+j],sx[t][g]));
                    else xq[t][12'(g*G+j)]=0;
                end
            end
        end
    endtask

    task automatic integer_projection;
        int signed sum;
        for (int r=0;r<rows;r++) begin
            for (int t=0;t<tokens;t++) begin
                result[r*B+t]=0.0;
                for (int g=0;g<(width+G-1)/G;g++) begin
                    sum=0;
                    for (int j=0;j<G;j++)
                        sum+=32'(wq[r][12'(g*G+j)])*32'(xq[t][12'(g*G+j)]);
                    group_acc[r][t][g]=sum;
                    result[r*B+t]+=(real'(sum)*sw[r][g])*sx[t][g];
                end
            end
        end
    endtask

    task automatic floating_projection;
        real partial[8];
        real sum;
        for (int r=0;r<rows;r++) begin
            for (int t=0;t<tokens;t++) begin
                sum=0.0;
                for (int l=0;l<8;l++) partial[l]=0.0;
                if (width%8==0) begin
                    for (int j=0;j<width;j+=8) begin
                        partial[0]+=weights[r*K+j+0]*x[t][j+0];
                        partial[1]+=weights[r*K+j+1]*x[t][j+1];
                        partial[2]+=weights[r*K+j+2]*x[t][j+2];
                        partial[3]+=weights[r*K+j+3]*x[t][j+3];
                        partial[4]+=weights[r*K+j+4]*x[t][j+4];
                        partial[5]+=weights[r*K+j+5]*x[t][j+5];
                        partial[6]+=weights[r*K+j+6]*x[t][j+6];
                        partial[7]+=weights[r*K+j+7]*x[t][j+7];
                    end
                    for (int l=0;l<8;l++) sum+=partial[l];
                end else begin
                    for (int j=0;j<width;j++) sum+=weights[r*K+j]*x[t][j];
                end
                float_result[r*B+t]=sum;
            end
        end
    endtask
    always @(posedge clk) begin
        real maximum, value, sum, scale, energy, error_energy, delta;
        real ref_ws, ref_xs;
        longint signed isum, qw, qx;
        if (start) begin
            if (rows<1 || rows>R || width<1 || width>K || tokens<1 || tokens>B)
                $fatal(1,"Invalid probe dimensions");
            case (operation)
                0: begin
                    // Real first-layer Q inputs, or explicitly synthetic stress inputs.
                    for (int t=0;t<tokens;t++) begin
                        sum=0.0;
                        if (fixture==1) begin
                            if (width!=1024) $fatal(1,"Embedding fixture width must be1024");
                            for (int j=0;j<width;j++) sum+=embedding[t*1024+j]*embedding[t*1024+j];
                            scale=1.0/$sqrt(sum/1024.0+1.0e-6);
                        end else scale=1.0;
                        for (int j=0;j<width;j++) begin
                            if (fixture==1) x[t][j]=embedding[t*1024+j]*scale*norm_weight[j];
                            else if (fixture==2) x[t][j]=0.0;
                            else x[t][j]=real'(((j*17+t*31)%255)-127)*0.03125;
                        end
                    end
                end
                1: begin
                    for (int r=0;r<rows;r++) begin
                        for (int g=0;g<(width+G-1)/G;g++) begin
                            maximum=0.0;
                            for (int j=0;j<G;j++) if (g*G+j<width) begin
                                value=weights[r*K+g*G+j];
                                if ((value-value)!=0.0) $fatal(1,"Nonfinite weight");
                                if (value<0.0) value=-value;
                                if (value>maximum) maximum=value;
                            end
                            sw[r][g]=maximum==0.0?1.0:maximum/127.0;
                            for (int j=0;j<G;j++) begin
                                if (g*G+j<width) wq[r][12'(g*G+j)]=8'(quantize(weights[r*K+g*G+j],sw[r][g]));
                                else wq[r][12'(g*G+j)]=0;
                            end
                        end
                    end
                end
                2: begin prepare_activations();integer_projection();end
                3: integer_projection();
                4: begin
                    failures=0;
                    for (int r=0;r<rows;r++) begin
                        for (int t=0;t<tokens;t++) begin
                            reference_result[r*B+t]=0.0;
                            for (int g=0;g<(width+G-1)/G;g++) begin
                                ref_ws=0.0;ref_xs=0.0;
                                for (int j=0;j<G;j++) if (g*G+j<width) begin
                                    value=weights[r*K+g*G+j];if(value<0.0)value=-value;
                                    if(value>ref_ws)ref_ws=value;
                                    value=x[t][g*G+j];if(value<0.0)value=-value;
                                    if(value>ref_xs)ref_xs=value;
                                end
                                ref_ws=ref_ws==0.0?1.0:ref_ws/127.0;
                                ref_xs=ref_xs==0.0?1.0:ref_xs/127.0;
                                if(ref_ws!=sw[r][g] || ref_xs!=sx[t][g]) failures++;
                                isum=0;
                                for (int j=0;j<G;j++) begin
                                    qw=0;qx=0;
                                    if(g*G+j<width) begin
                                        qw=quantize_reference(weights[r*K+g*G+j],ref_ws);
                                        qx=quantize_reference(x[t][g*G+j],ref_xs);
                                    end
                                    if(qw!=64'(wq[r][12'(g*G+j)]) || qx!=64'(xq[t][12'(g*G+j)])) failures++;
                                    isum+=qw*qx;
                                end
                                if(isum!=64'(group_acc[r][t][g])) failures++;
                                reference_result[r*B+t]+=(real'(isum)*ref_ws)*ref_xs;
                            end
                            if(reference_result[r*B+t]!=result[r*B+t]) failures++;
                        end
                    end
                end
                5: begin
                    floating_projection();
                    max_error=0.0;error_energy=0.0;energy=0.0;
                    for (int r=0;r<rows;r++) begin
                        for (int t=0;t<tokens;t++) begin
                            sum=float_result[r*B+t];
                            delta=result[r*B+t]-sum;
                            error_energy+=delta*delta;energy+=sum*sum;
                            if(delta<0.0)delta=-delta;
                            if(delta>max_error)max_error=delta;
                        end
                    end
                    relative_rmse=energy==0.0?0.0:$sqrt(error_energy/energy);
                end
                6: prepare_activations();
                8: floating_projection();
                9: begin end // Unmodified production worker executes on this edge.
                10: begin
                    failures=0;
                    for(int r=0;r<rows;r++) for(int t=0;t<tokens;t++) begin
                        value=tokens==1?worker_result[r]:worker_batch_result[r*B+t];
                        if(value!=float_result[r*B+t]) failures++;
                    end
                end
                7: begin
                    // Known answers, including ties, signed extrema and saturation.
                    failures=0;
                    if(quantize(0.0,1.0)!=0 || quantize(-0.0,1.0)!=0) failures++;
                    if(quantize(0.5,1.0)!=0 || quantize(-0.5,1.0)!=0) failures++;
                    if(quantize(1.5,1.0)!=2 || quantize(-1.5,1.0)!=-2) failures++;
                    if(quantize(2.5,1.0)!=2 || quantize(-2.5,1.0)!=-2) failures++;
                    if(quantize(126.5,1.0)!=126 || quantize(-126.5,1.0)!=-126) failures++;
                    if(quantize(128.0,1.0)!=127 || quantize(-128.0,1.0)!=-127) failures++;
                    if(quantize(3.0,2.0)!=2 || quantize(-5.0,2.0)!=-2) failures++;
                    if(32'(-8'sd127)*32'(-8'sd127)!=16129) failures++;
                end
                default: $fatal(1,"Unknown probe operation");
            endcase
        end
    end
endmodule
