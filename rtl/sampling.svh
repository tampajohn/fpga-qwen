// Chat decoding: temperature 0.7, top-k 20, top-p 0.8, optional presence penalty.
// Randomness is an external uniform variate; all token-selection math is SV.
function automatic int sample_chat_token(input real raw_logits[], input int history[],
                                         input int history_length, input real uniform_value,
                                         input real presence_penalty);
    real top_scores[20], probability[20];
    int top_ids[20];
    bit seen[];
    real candidate, total, retained, target, cumulative;
    int count, kept, chosen, first;
    seen=new[raw_logits.size()];
    first=history_length>256 ? history_length-256 : 0;
    for (int t=first;t<history_length;t++)
        if (history[t]>=0 && history[t]<raw_logits.size() && history[t]<151643)
            seen[history[t]]=1;
    for (int j=0;j<20;j++) begin top_scores[j]=-1.0e300;top_ids[j]=-1;end
    count=raw_logits.size()<20 ? raw_logits.size() : 20;
    for (int token=0;token<raw_logits.size();token++) begin
        candidate=(raw_logits[token]-(seen[token] ? presence_penalty : 0.0))/0.7;
        for (int rank=0;rank<count;rank++) begin
            if (candidate>top_scores[rank]) begin
                for (int j=count-1;j>rank;j--) begin
                    top_scores[j]=top_scores[j-1];top_ids[j]=top_ids[j-1];
                end
                top_scores[rank]=candidate;top_ids[rank]=token;
                break;
            end
        end
    end
    total=0.0;
    for (int j=0;j<count;j++) begin
        probability[j]=$exp(top_scores[j]-top_scores[0]);total+=probability[j];
    end
    retained=0.0;kept=0;
    for (int j=0;j<count;j++) begin
        retained+=probability[j];kept=j+1;
        if (retained>=0.8*total) break;
    end
    target=uniform_value*retained;cumulative=0.0;chosen=top_ids[kept-1];
    for (int j=0;j<kept;j++) begin
        cumulative+=probability[j];
        if (target<cumulative) begin chosen=top_ids[j];break;end
    end
    return chosen;
endfunction
