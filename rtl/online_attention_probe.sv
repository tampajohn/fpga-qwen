// Isolated FP64 tiled online-softmax experiment. Not a model backend.
// Algorithm reference: Dao et al., arXiv:2205.14135, Algorithm 1.
// Original implementation; all attention arithmetic runs in SystemVerilog.
module online_attention_probe;
    localparam int Q=4, D=16, N=65, TILE=16;
    real queries[Q][D], keys[N][D], values[N][D];
    real scores[Q][TILE], output_sum[Q][D], maximum[Q], normalizer[Q];
    real reference_output[Q][D], reference_scores[N];
    real dot, row_max, tile_max, next_max, denominator, probability, rescale;
    real factor;
    int lengths[6]='{1,15,16,17,64,65};
    int length, visible, used;

    initial begin
        for (int scenario=0;scenario<3;scenario++) begin
            factor=scenario==1 ? 64.0 : scenario==2 ? 0.0009765625 : 1.0;
            for (int q=0;q<Q;q++)
                for (int j=0;j<D;j++) queries[q][j]=real'((q*13+j*7)%19-9)/16.0*factor;
            for (int t=0;t<N;t++)
                for (int j=0;j<D;j++) begin
                    keys[t][j]=real'((t*11+j*3)%23-11)/16.0*factor;
                    values[t][j]=real'((t*5+j*17)%29-14)/8.0;
                end
            for (int test=0;test<6;test++) begin
                length=lengths[test];
                // Existing-style reference: full scores, global softmax, then V.
                for (int q=0;q<Q;q++) begin
                    visible=length-Q+q+1;
                    if (visible<1) visible=1;
                    row_max=-1.0e300;
                    for (int t=0;t<visible;t++) begin
                        dot=0.0;
                        for (int j=0;j<D;j++) dot+=queries[q][j]*keys[t][j];
                        reference_scores[t]=dot/4.0;
                        if (reference_scores[t]>row_max) row_max=reference_scores[t];
                    end
                    denominator=0.0;
                    for (int t=0;t<visible;t++) begin
                        reference_scores[t]=$exp(reference_scores[t]-row_max);
                        denominator+=reference_scores[t];
                    end
                    for (int j=0;j<D;j++) reference_output[q][j]=0.0;
                    for (int t=0;t<visible;t++)
                        for (int j=0;j<D;j++)
                            reference_output[q][j]+=(reference_scores[t]/denominator)*values[t][j];
                end
                for (int q=0;q<Q;q++) begin
                    maximum[q]=-1.0e300;normalizer[q]=0.0;
                    for (int j=0;j<D;j++) output_sum[q][j]=0.0;
                end
                // Only a Q x TILE score buffer; output stays unnormalized until end.
                for (int base=0;base<length;base+=TILE) begin
                    for (int q=0;q<Q;q++) begin
                        visible=length-Q+q+1;
                        if (visible<1) visible=1;
                        used=visible-base;
                        if (used>TILE) used=TILE;
                        if (used>0) begin
                            tile_max=-1.0e300;
                            for (int t=0;t<used;t++) begin
                                dot=0.0;
                                for (int j=0;j<D;j++) dot+=queries[q][j]*keys[base+t][j];
                                scores[q][t]=dot/4.0;
                                if (scores[q][t]>tile_max) tile_max=scores[q][t];
                            end
                            next_max=maximum[q]>tile_max ? maximum[q] : tile_max;
                            rescale=normalizer[q]==0.0 ? 0.0 : $exp(maximum[q]-next_max);
                            normalizer[q]*=rescale;
                            for (int j=0;j<D;j++) output_sum[q][j]*=rescale;
                            for (int t=0;t<used;t++) begin
                                probability=$exp(scores[q][t]-next_max);
                                normalizer[q]+=probability;
                                for (int j=0;j<D;j++) output_sum[q][j]+=probability*values[base+t][j];
                            end
                            maximum[q]=next_max;
                        end
                    end
                end
                for (int q=0;q<Q;q++)
                    for (int j=0;j<D;j++)
                        $display("RESULT %0d %0d %0d %0d %.17g %.17g",scenario,length,q,j,
                                 reference_output[q][j],output_sum[q][j]/normalizer[q]);
            end
        end
        $finish;
    end
endmodule
