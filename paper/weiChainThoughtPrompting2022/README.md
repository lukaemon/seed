## Log
### 20230117

### 20230116 `GSM8K`
- `main` subset lays out the rationale objectively. 
  - `socratic` subset carries on a series of self qa subjectively. 
  -  Prompt in the paper is more akin to `main` style. 
- When `num_beams=4`, answer = **Janet eats 3 + 4 = 7 eggs a day. She sells 16 - 7 = 11 eggs a day at the farmers' market. She makes 11 * 2 = 22 dollars a day at the farmers' market. The answer is 22.**
  - Even though the answer is wrong. The miss step is 16-7=9, not 11. The logic overall is fine. This is the emergent part of the LLM for reasoning task. So many little intermediate step could go wrong, and on wrong would render the final answer wrong.   
  - It's actually not emergent. To get one reasoning task right, observed from final answer perspective, it's a quantum leap wrt to modele size. However, observed from rationale perpsective, it's not quantum leap and magical anymore. You can observe little mistakes here and there, and they lead to wrong answer. 
  - The other mistake is Janet won't eat 7 eggs, but rather use 7 per day. Semantically, this 3b model got it wrong, even though the number, 7, is the right number for next stage calculation. 
  - Also you can see the beauty of such thinking out loud rationale engineering. It makes abstract computation transparent and **readable** so you could understand and debug incrementally, rather than betting on pure wishful thinking of quantum leap. 
  - When I switch off `num_beams` to `greedy decoding`, it actually generate right answer at 3b scale: **She eats 3 + 4 = 7 eggs every day. She has 16 - 7 = 9 eggs left. She sells 9 * 2 = $18 at the farmers' market. The answer is 18.** Simply amazing. This is better than normal 5 year old reasoning lol. 