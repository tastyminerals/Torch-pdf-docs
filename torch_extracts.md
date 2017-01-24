# Torch docs extracts
### nn.Identity()
This module returns whatever input it receives without any changes to the input.
It is generally used with nn.ParallelTable which applies its i-th module to i-th input.
nn.ParallelTable expects to have a table such that the number of its elements is equal to the number of modules. 
So, when you don't want to do anything to x-th input, you use nn.Identity().

### nn.ParallelTable()
nn.ParallelTable applies its i-th module to i-th elements of the input table. For example,

p = nn.ParallelTable()
p:add(nn.Linear(2,1))
p:add(nn.Identity())

-- notice that the number of x elements is equal to the number of modules in p
x = {}
x[1] = torch.rand(2)
x[2] = torch.rand(5)

p:forward(x)

### nn.CAddTable()
Just adds together several Tensors, e.g. {1,1,1} + {2,2,2} = {3,3,3}.

### nn.NarrowTable(offset [, length])
Takes a table as input and outputs a subtable starting at index offset.
{2x3, 1x2, 5x5} --> nn.NarrowTable(2) --> {1x2}
{2x3, 1x2, 5x5, 3x1} --> nn.NarrowTable(2,4) --> {1x2, 5x5, 3x1}

### nn.SelectTable(idx)
Selects an idx element of the table.
idx can be int or string, positive or negative.

### nn.ConcatTable()
It's a container module which applies each member module to the same input Tensor/table.
For example, you have input x and 3 members in nn.ConcatTable.
First, nn.ConcatTable will apply 1-th member to x input, then 2-th member and finally 3-d to the same input x.

### nn.CMulTable()
Element-wise multiplication. Accepts a table of Tensors.
