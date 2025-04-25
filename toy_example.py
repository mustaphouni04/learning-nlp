import torch
import torch.nn.functional as F

tens = torch.arange(0,50258, dtype=torch.float32).unsqueeze(0)
val, ind = torch.topk(tens, 50)
filtered = torch.full_like(tens,float('-inf'))
final = filtered.scatter_(1,ind,val)

print(final[:,0:5])
print(final[:,50200:])

val, ind = torch.sort(final, descending=True)
print(val,ind)

cumulative_probs = torch.cumsum(F.softmax(val, dim=-1),dim=-1)
print(cumulative_probs)


