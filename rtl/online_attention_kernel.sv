// Original isolated attention experiment; not a model/backend implementation.
module online_attention_kernel(
    input bit clk, input int method, length, count,
    input real queries[1024], keys[524288], values[524288],
    output real result[1024]
);
    localparam int D=128, Q=8, TILE=16;
    real pair_scores[4096][2], scratch[128];
    real tile_scores[16][8], accum[128][8], maxima[8], denom[8];
    real sums[2], pair_max[2], total, scale, key, dot;
    real tile_max, next_max, rescale, probability;
    int visible, used;
    always @(posedge clk) begin
      if(length<count || length>4096 || count<1 || count>8 || method<0 || method>1)
        $fatal(1,"Invalid kernel shape/method");
      if(method==0) begin
        // Adapted from original project op5 QK2: one head, contiguous KV,
        // feature-major Q/output. Each query preserves its original j/t order.
        for(int query=0;query<count;query+=2) begin
          pair_max[0]=-1.0e300; pair_max[1]=-1.0e300;
          for(int t=0;t<=length-count+query;t++) begin
            sums[0]=0.0;sums[1]=0.0;
            for(int j=0;j<D;j++) begin
              key=keys[t*D+j];
              sums[0]+=queries[j*Q+query]*key;
              sums[1]+=queries[j*Q+query+1]*key;
            end
            pair_scores[t][0]=sums[0]/$sqrt(128.0);
            pair_scores[t][1]=sums[1]/$sqrt(128.0);
            if(pair_scores[t][0]>pair_max[0]) pair_max[0]=pair_scores[t][0];
            if(pair_scores[t][1]>pair_max[1]) pair_max[1]=pair_scores[t][1];
          end
          if(query+1<count) begin
            sums[1]=0.0;
            for(int j=0;j<D;j++) sums[1]+=queries[j*Q+query+1]*keys[(length-count+query+1)*D+j];
            pair_scores[length-count+query+1][1]=sums[1]/$sqrt(128.0);
            if(pair_scores[length-count+query+1][1]>pair_max[1]) pair_max[1]=pair_scores[length-count+query+1][1];
          end
          for(int lane=0;lane<2 && query+lane<count;lane++) begin
            total=0.0;
            for(int t=0;t<=length-count+query+lane;t++) begin
              pair_scores[t][lane]=$exp(pair_scores[t][lane]-pair_max[lane]);
              total+=pair_scores[t][lane];
            end
            for(int j=0;j<D;j++) scratch[j]=0.0;
            for(int t=0;t<=length-count+query+lane;t++) begin
              scale=pair_scores[t][lane]/total;
              for(int j=0;j<D;j++) scratch[j]+=scale*values[t*D+j];
            end
            for(int j=0;j<D;j++) result[j*Q+query+lane]=scratch[j];
          end
        end
      end else begin
        for(int q=0;q<count;q++) begin
          maxima[q]=-1.0e300;denom[q]=0.0;
          for(int j=0;j<D;j++) accum[j][q]=0.0;
        end
        for(int base=0;base<length;base+=TILE) begin
          // Scores keep ascending feature accumulation, independent queries.
          for(int t=0;t<TILE && base+t<length;t++) begin
`ifdef ONLINE_QK2
            // Independent query pair stays in local sums across all features.
            // Same ascending feature order; no horizontal reassociation.
            for(int q=0;q<count;q+=2) begin
              sums[0]=0.0;sums[1]=0.0;
              for(int j=0;j<D;j++) begin
                key=keys[(base+t)*D+j];
                sums[0]+=queries[j*Q+q]*key;
                sums[1]+=queries[j*Q+q+1]*key;
              end
              tile_scores[t][q]=sums[0]/$sqrt(128.0);
              if(q+1<count) tile_scores[t][q+1]=sums[1]/$sqrt(128.0);
            end
`else
            for(int q=0;q<count;q++) tile_scores[t][q]=0.0;
            for(int j=0;j<D;j++) begin
              key=keys[(base+t)*D+j];
              for(int q=0;q<count;q++) tile_scores[t][q]+=queries[j*Q+q]*key;
            end
            for(int q=0;q<count;q++) tile_scores[t][q]/=$sqrt(128.0);
`endif
          end
          for(int q=0;q<count;q++) begin
            visible=length-count+q+1;used=visible-base;
            if(used>TILE) used=TILE;
            if(used>0) begin
              tile_max=-1.0e300;
              for(int t=0;t<used;t++) if(tile_scores[t][q]>tile_max) tile_max=tile_scores[t][q];
              next_max=maxima[q]>tile_max ? maxima[q] : tile_max;
              rescale=denom[q]==0.0 ? 0.0 : $exp(maxima[q]-next_max);
              denom[q]*=rescale;
              for(int j=0;j<D;j++) accum[j][q]*=rescale;
              for(int t=0;t<used;t++) begin
                probability=$exp(tile_scores[t][q]-next_max);
                denom[q]+=probability;
                for(int j=0;j<D;j++) accum[j][q]+=probability*values[(base+t)*D+j];
              end
              maxima[q]=next_max;
            end
          end
        end
        for(int q=0;q<count;q++) for(int j=0;j<D;j++) result[j*Q+q]=accum[j][q]/denom[q];
      end
    end
endmodule
