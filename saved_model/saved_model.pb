��
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

$
DisableCopyOnRead
resource�
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-rc1-8-g6887368d6d48��
v
countVarHandleOp*
_output_shapes
: *

debug_namecount/*
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
v
totalVarHandleOp*
_output_shapes
: *

debug_nametotal/*
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
|
count_1VarHandleOp*
_output_shapes
: *

debug_name
count_1/*
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
total_1VarHandleOp*
_output_shapes
: *

debug_name
total_1/*
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
�
v/final/biasVarHandleOp*
_output_shapes
: *

debug_namev/final/bias/*
dtype0*
shape:*
shared_namev/final/bias
i
 v/final/bias/Read/ReadVariableOpReadVariableOpv/final/bias*
_output_shapes
:*
dtype0
�
m/final/biasVarHandleOp*
_output_shapes
: *

debug_namem/final/bias/*
dtype0*
shape:*
shared_namem/final/bias
i
 m/final/bias/Read/ReadVariableOpReadVariableOpm/final/bias*
_output_shapes
:*
dtype0
�
v/final/kernelVarHandleOp*
_output_shapes
: *

debug_namev/final/kernel/*
dtype0*
shape:	�*
shared_namev/final/kernel
r
"v/final/kernel/Read/ReadVariableOpReadVariableOpv/final/kernel*
_output_shapes
:	�*
dtype0
�
m/final/kernelVarHandleOp*
_output_shapes
: *

debug_namem/final/kernel/*
dtype0*
shape:	�*
shared_namem/final/kernel
r
"m/final/kernel/Read/ReadVariableOpReadVariableOpm/final/kernel*
_output_shapes
:	�*
dtype0
�
v/dense1/biasVarHandleOp*
_output_shapes
: *

debug_namev/dense1/bias/*
dtype0*
shape:�*
shared_namev/dense1/bias
l
!v/dense1/bias/Read/ReadVariableOpReadVariableOpv/dense1/bias*
_output_shapes	
:�*
dtype0
�
m/dense1/biasVarHandleOp*
_output_shapes
: *

debug_namem/dense1/bias/*
dtype0*
shape:�*
shared_namem/dense1/bias
l
!m/dense1/bias/Read/ReadVariableOpReadVariableOpm/dense1/bias*
_output_shapes	
:�*
dtype0
�
v/dense1/kernelVarHandleOp*
_output_shapes
: * 

debug_namev/dense1/kernel/*
dtype0*
shape:���* 
shared_namev/dense1/kernel
v
#v/dense1/kernel/Read/ReadVariableOpReadVariableOpv/dense1/kernel*!
_output_shapes
:���*
dtype0
�
m/dense1/kernelVarHandleOp*
_output_shapes
: * 

debug_namem/dense1/kernel/*
dtype0*
shape:���* 
shared_namem/dense1/kernel
v
#m/dense1/kernel/Read/ReadVariableOpReadVariableOpm/dense1/kernel*!
_output_shapes
:���*
dtype0
�
v/block5_conv3/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block5_conv3/bias/*
dtype0*
shape:�*$
shared_namev/block5_conv3/bias
x
'v/block5_conv3/bias/Read/ReadVariableOpReadVariableOpv/block5_conv3/bias*
_output_shapes	
:�*
dtype0
�
m/block5_conv3/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block5_conv3/bias/*
dtype0*
shape:�*$
shared_namem/block5_conv3/bias
x
'm/block5_conv3/bias/Read/ReadVariableOpReadVariableOpm/block5_conv3/bias*
_output_shapes	
:�*
dtype0
�
v/block5_conv3/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block5_conv3/kernel/*
dtype0*
shape:��*&
shared_namev/block5_conv3/kernel
�
)v/block5_conv3/kernel/Read/ReadVariableOpReadVariableOpv/block5_conv3/kernel*(
_output_shapes
:��*
dtype0
�
m/block5_conv3/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block5_conv3/kernel/*
dtype0*
shape:��*&
shared_namem/block5_conv3/kernel
�
)m/block5_conv3/kernel/Read/ReadVariableOpReadVariableOpm/block5_conv3/kernel*(
_output_shapes
:��*
dtype0
�
v/block5_conv2/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block5_conv2/bias/*
dtype0*
shape:�*$
shared_namev/block5_conv2/bias
x
'v/block5_conv2/bias/Read/ReadVariableOpReadVariableOpv/block5_conv2/bias*
_output_shapes	
:�*
dtype0
�
m/block5_conv2/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block5_conv2/bias/*
dtype0*
shape:�*$
shared_namem/block5_conv2/bias
x
'm/block5_conv2/bias/Read/ReadVariableOpReadVariableOpm/block5_conv2/bias*
_output_shapes	
:�*
dtype0
�
v/block5_conv2/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block5_conv2/kernel/*
dtype0*
shape:��*&
shared_namev/block5_conv2/kernel
�
)v/block5_conv2/kernel/Read/ReadVariableOpReadVariableOpv/block5_conv2/kernel*(
_output_shapes
:��*
dtype0
�
m/block5_conv2/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block5_conv2/kernel/*
dtype0*
shape:��*&
shared_namem/block5_conv2/kernel
�
)m/block5_conv2/kernel/Read/ReadVariableOpReadVariableOpm/block5_conv2/kernel*(
_output_shapes
:��*
dtype0
�
v/block5_conv1/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block5_conv1/bias/*
dtype0*
shape:�*$
shared_namev/block5_conv1/bias
x
'v/block5_conv1/bias/Read/ReadVariableOpReadVariableOpv/block5_conv1/bias*
_output_shapes	
:�*
dtype0
�
m/block5_conv1/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block5_conv1/bias/*
dtype0*
shape:�*$
shared_namem/block5_conv1/bias
x
'm/block5_conv1/bias/Read/ReadVariableOpReadVariableOpm/block5_conv1/bias*
_output_shapes	
:�*
dtype0
�
v/block5_conv1/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block5_conv1/kernel/*
dtype0*
shape:��*&
shared_namev/block5_conv1/kernel
�
)v/block5_conv1/kernel/Read/ReadVariableOpReadVariableOpv/block5_conv1/kernel*(
_output_shapes
:��*
dtype0
�
m/block5_conv1/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block5_conv1/kernel/*
dtype0*
shape:��*&
shared_namem/block5_conv1/kernel
�
)m/block5_conv1/kernel/Read/ReadVariableOpReadVariableOpm/block5_conv1/kernel*(
_output_shapes
:��*
dtype0
�
v/block4_conv3/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block4_conv3/bias/*
dtype0*
shape:�*$
shared_namev/block4_conv3/bias
x
'v/block4_conv3/bias/Read/ReadVariableOpReadVariableOpv/block4_conv3/bias*
_output_shapes	
:�*
dtype0
�
m/block4_conv3/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block4_conv3/bias/*
dtype0*
shape:�*$
shared_namem/block4_conv3/bias
x
'm/block4_conv3/bias/Read/ReadVariableOpReadVariableOpm/block4_conv3/bias*
_output_shapes	
:�*
dtype0
�
v/block4_conv3/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block4_conv3/kernel/*
dtype0*
shape:��*&
shared_namev/block4_conv3/kernel
�
)v/block4_conv3/kernel/Read/ReadVariableOpReadVariableOpv/block4_conv3/kernel*(
_output_shapes
:��*
dtype0
�
m/block4_conv3/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block4_conv3/kernel/*
dtype0*
shape:��*&
shared_namem/block4_conv3/kernel
�
)m/block4_conv3/kernel/Read/ReadVariableOpReadVariableOpm/block4_conv3/kernel*(
_output_shapes
:��*
dtype0
�
v/block4_conv2/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block4_conv2/bias/*
dtype0*
shape:�*$
shared_namev/block4_conv2/bias
x
'v/block4_conv2/bias/Read/ReadVariableOpReadVariableOpv/block4_conv2/bias*
_output_shapes	
:�*
dtype0
�
m/block4_conv2/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block4_conv2/bias/*
dtype0*
shape:�*$
shared_namem/block4_conv2/bias
x
'm/block4_conv2/bias/Read/ReadVariableOpReadVariableOpm/block4_conv2/bias*
_output_shapes	
:�*
dtype0
�
v/block4_conv2/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block4_conv2/kernel/*
dtype0*
shape:��*&
shared_namev/block4_conv2/kernel
�
)v/block4_conv2/kernel/Read/ReadVariableOpReadVariableOpv/block4_conv2/kernel*(
_output_shapes
:��*
dtype0
�
m/block4_conv2/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block4_conv2/kernel/*
dtype0*
shape:��*&
shared_namem/block4_conv2/kernel
�
)m/block4_conv2/kernel/Read/ReadVariableOpReadVariableOpm/block4_conv2/kernel*(
_output_shapes
:��*
dtype0
�
v/block4_conv1/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block4_conv1/bias/*
dtype0*
shape:�*$
shared_namev/block4_conv1/bias
x
'v/block4_conv1/bias/Read/ReadVariableOpReadVariableOpv/block4_conv1/bias*
_output_shapes	
:�*
dtype0
�
m/block4_conv1/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block4_conv1/bias/*
dtype0*
shape:�*$
shared_namem/block4_conv1/bias
x
'm/block4_conv1/bias/Read/ReadVariableOpReadVariableOpm/block4_conv1/bias*
_output_shapes	
:�*
dtype0
�
v/block4_conv1/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block4_conv1/kernel/*
dtype0*
shape:��*&
shared_namev/block4_conv1/kernel
�
)v/block4_conv1/kernel/Read/ReadVariableOpReadVariableOpv/block4_conv1/kernel*(
_output_shapes
:��*
dtype0
�
m/block4_conv1/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block4_conv1/kernel/*
dtype0*
shape:��*&
shared_namem/block4_conv1/kernel
�
)m/block4_conv1/kernel/Read/ReadVariableOpReadVariableOpm/block4_conv1/kernel*(
_output_shapes
:��*
dtype0
�
v/block3_conv3/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block3_conv3/bias/*
dtype0*
shape:�*$
shared_namev/block3_conv3/bias
x
'v/block3_conv3/bias/Read/ReadVariableOpReadVariableOpv/block3_conv3/bias*
_output_shapes	
:�*
dtype0
�
m/block3_conv3/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block3_conv3/bias/*
dtype0*
shape:�*$
shared_namem/block3_conv3/bias
x
'm/block3_conv3/bias/Read/ReadVariableOpReadVariableOpm/block3_conv3/bias*
_output_shapes	
:�*
dtype0
�
v/block3_conv3/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block3_conv3/kernel/*
dtype0*
shape:��*&
shared_namev/block3_conv3/kernel
�
)v/block3_conv3/kernel/Read/ReadVariableOpReadVariableOpv/block3_conv3/kernel*(
_output_shapes
:��*
dtype0
�
m/block3_conv3/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block3_conv3/kernel/*
dtype0*
shape:��*&
shared_namem/block3_conv3/kernel
�
)m/block3_conv3/kernel/Read/ReadVariableOpReadVariableOpm/block3_conv3/kernel*(
_output_shapes
:��*
dtype0
�
v/block3_conv2/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block3_conv2/bias/*
dtype0*
shape:�*$
shared_namev/block3_conv2/bias
x
'v/block3_conv2/bias/Read/ReadVariableOpReadVariableOpv/block3_conv2/bias*
_output_shapes	
:�*
dtype0
�
m/block3_conv2/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block3_conv2/bias/*
dtype0*
shape:�*$
shared_namem/block3_conv2/bias
x
'm/block3_conv2/bias/Read/ReadVariableOpReadVariableOpm/block3_conv2/bias*
_output_shapes	
:�*
dtype0
�
v/block3_conv2/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block3_conv2/kernel/*
dtype0*
shape:��*&
shared_namev/block3_conv2/kernel
�
)v/block3_conv2/kernel/Read/ReadVariableOpReadVariableOpv/block3_conv2/kernel*(
_output_shapes
:��*
dtype0
�
m/block3_conv2/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block3_conv2/kernel/*
dtype0*
shape:��*&
shared_namem/block3_conv2/kernel
�
)m/block3_conv2/kernel/Read/ReadVariableOpReadVariableOpm/block3_conv2/kernel*(
_output_shapes
:��*
dtype0
�
v/block3_conv1/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block3_conv1/bias/*
dtype0*
shape:�*$
shared_namev/block3_conv1/bias
x
'v/block3_conv1/bias/Read/ReadVariableOpReadVariableOpv/block3_conv1/bias*
_output_shapes	
:�*
dtype0
�
m/block3_conv1/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block3_conv1/bias/*
dtype0*
shape:�*$
shared_namem/block3_conv1/bias
x
'm/block3_conv1/bias/Read/ReadVariableOpReadVariableOpm/block3_conv1/bias*
_output_shapes	
:�*
dtype0
�
v/block3_conv1/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block3_conv1/kernel/*
dtype0*
shape:��*&
shared_namev/block3_conv1/kernel
�
)v/block3_conv1/kernel/Read/ReadVariableOpReadVariableOpv/block3_conv1/kernel*(
_output_shapes
:��*
dtype0
�
m/block3_conv1/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block3_conv1/kernel/*
dtype0*
shape:��*&
shared_namem/block3_conv1/kernel
�
)m/block3_conv1/kernel/Read/ReadVariableOpReadVariableOpm/block3_conv1/kernel*(
_output_shapes
:��*
dtype0
�
v/block2_conv2/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block2_conv2/bias/*
dtype0*
shape:�*$
shared_namev/block2_conv2/bias
x
'v/block2_conv2/bias/Read/ReadVariableOpReadVariableOpv/block2_conv2/bias*
_output_shapes	
:�*
dtype0
�
m/block2_conv2/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block2_conv2/bias/*
dtype0*
shape:�*$
shared_namem/block2_conv2/bias
x
'm/block2_conv2/bias/Read/ReadVariableOpReadVariableOpm/block2_conv2/bias*
_output_shapes	
:�*
dtype0
�
v/block2_conv2/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block2_conv2/kernel/*
dtype0*
shape:��*&
shared_namev/block2_conv2/kernel
�
)v/block2_conv2/kernel/Read/ReadVariableOpReadVariableOpv/block2_conv2/kernel*(
_output_shapes
:��*
dtype0
�
m/block2_conv2/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block2_conv2/kernel/*
dtype0*
shape:��*&
shared_namem/block2_conv2/kernel
�
)m/block2_conv2/kernel/Read/ReadVariableOpReadVariableOpm/block2_conv2/kernel*(
_output_shapes
:��*
dtype0
�
v/block2_conv1/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block2_conv1/bias/*
dtype0*
shape:�*$
shared_namev/block2_conv1/bias
x
'v/block2_conv1/bias/Read/ReadVariableOpReadVariableOpv/block2_conv1/bias*
_output_shapes	
:�*
dtype0
�
m/block2_conv1/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block2_conv1/bias/*
dtype0*
shape:�*$
shared_namem/block2_conv1/bias
x
'm/block2_conv1/bias/Read/ReadVariableOpReadVariableOpm/block2_conv1/bias*
_output_shapes	
:�*
dtype0
�
v/block2_conv1/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block2_conv1/kernel/*
dtype0*
shape:@�*&
shared_namev/block2_conv1/kernel
�
)v/block2_conv1/kernel/Read/ReadVariableOpReadVariableOpv/block2_conv1/kernel*'
_output_shapes
:@�*
dtype0
�
m/block2_conv1/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block2_conv1/kernel/*
dtype0*
shape:@�*&
shared_namem/block2_conv1/kernel
�
)m/block2_conv1/kernel/Read/ReadVariableOpReadVariableOpm/block2_conv1/kernel*'
_output_shapes
:@�*
dtype0
�
v/block1_conv2/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block1_conv2/bias/*
dtype0*
shape:@*$
shared_namev/block1_conv2/bias
w
'v/block1_conv2/bias/Read/ReadVariableOpReadVariableOpv/block1_conv2/bias*
_output_shapes
:@*
dtype0
�
m/block1_conv2/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block1_conv2/bias/*
dtype0*
shape:@*$
shared_namem/block1_conv2/bias
w
'm/block1_conv2/bias/Read/ReadVariableOpReadVariableOpm/block1_conv2/bias*
_output_shapes
:@*
dtype0
�
v/block1_conv2/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block1_conv2/kernel/*
dtype0*
shape:@@*&
shared_namev/block1_conv2/kernel
�
)v/block1_conv2/kernel/Read/ReadVariableOpReadVariableOpv/block1_conv2/kernel*&
_output_shapes
:@@*
dtype0
�
m/block1_conv2/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block1_conv2/kernel/*
dtype0*
shape:@@*&
shared_namem/block1_conv2/kernel
�
)m/block1_conv2/kernel/Read/ReadVariableOpReadVariableOpm/block1_conv2/kernel*&
_output_shapes
:@@*
dtype0
�
v/block1_conv1/biasVarHandleOp*
_output_shapes
: *$

debug_namev/block1_conv1/bias/*
dtype0*
shape:@*$
shared_namev/block1_conv1/bias
w
'v/block1_conv1/bias/Read/ReadVariableOpReadVariableOpv/block1_conv1/bias*
_output_shapes
:@*
dtype0
�
m/block1_conv1/biasVarHandleOp*
_output_shapes
: *$

debug_namem/block1_conv1/bias/*
dtype0*
shape:@*$
shared_namem/block1_conv1/bias
w
'm/block1_conv1/bias/Read/ReadVariableOpReadVariableOpm/block1_conv1/bias*
_output_shapes
:@*
dtype0
�
v/block1_conv1/kernelVarHandleOp*
_output_shapes
: *&

debug_namev/block1_conv1/kernel/*
dtype0*
shape:@*&
shared_namev/block1_conv1/kernel
�
)v/block1_conv1/kernel/Read/ReadVariableOpReadVariableOpv/block1_conv1/kernel*&
_output_shapes
:@*
dtype0
�
m/block1_conv1/kernelVarHandleOp*
_output_shapes
: *&

debug_namem/block1_conv1/kernel/*
dtype0*
shape:@*&
shared_namem/block1_conv1/kernel
�
)m/block1_conv1/kernel/Read/ReadVariableOpReadVariableOpm/block1_conv1/kernel*&
_output_shapes
:@*
dtype0
�
learning_rateVarHandleOp*
_output_shapes
: *

debug_namelearning_rate/*
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
�
	iterationVarHandleOp*
_output_shapes
: *

debug_name
iteration/*
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
�
block5_conv3/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock5_conv3/bias/*
dtype0*
shape:�*"
shared_nameblock5_conv3/bias
t
%block5_conv3/bias/Read/ReadVariableOpReadVariableOpblock5_conv3/bias*
_output_shapes	
:�*
dtype0
�
block5_conv3/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock5_conv3/kernel/*
dtype0*
shape:��*$
shared_nameblock5_conv3/kernel
�
'block5_conv3/kernel/Read/ReadVariableOpReadVariableOpblock5_conv3/kernel*(
_output_shapes
:��*
dtype0
�
block5_conv2/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock5_conv2/bias/*
dtype0*
shape:�*"
shared_nameblock5_conv2/bias
t
%block5_conv2/bias/Read/ReadVariableOpReadVariableOpblock5_conv2/bias*
_output_shapes	
:�*
dtype0
�
block5_conv2/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock5_conv2/kernel/*
dtype0*
shape:��*$
shared_nameblock5_conv2/kernel
�
'block5_conv2/kernel/Read/ReadVariableOpReadVariableOpblock5_conv2/kernel*(
_output_shapes
:��*
dtype0
�
block5_conv1/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock5_conv1/bias/*
dtype0*
shape:�*"
shared_nameblock5_conv1/bias
t
%block5_conv1/bias/Read/ReadVariableOpReadVariableOpblock5_conv1/bias*
_output_shapes	
:�*
dtype0
�
block5_conv1/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock5_conv1/kernel/*
dtype0*
shape:��*$
shared_nameblock5_conv1/kernel
�
'block5_conv1/kernel/Read/ReadVariableOpReadVariableOpblock5_conv1/kernel*(
_output_shapes
:��*
dtype0
�
block4_conv3/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock4_conv3/bias/*
dtype0*
shape:�*"
shared_nameblock4_conv3/bias
t
%block4_conv3/bias/Read/ReadVariableOpReadVariableOpblock4_conv3/bias*
_output_shapes	
:�*
dtype0
�
block4_conv3/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock4_conv3/kernel/*
dtype0*
shape:��*$
shared_nameblock4_conv3/kernel
�
'block4_conv3/kernel/Read/ReadVariableOpReadVariableOpblock4_conv3/kernel*(
_output_shapes
:��*
dtype0
�
block4_conv2/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock4_conv2/bias/*
dtype0*
shape:�*"
shared_nameblock4_conv2/bias
t
%block4_conv2/bias/Read/ReadVariableOpReadVariableOpblock4_conv2/bias*
_output_shapes	
:�*
dtype0
�
block4_conv2/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock4_conv2/kernel/*
dtype0*
shape:��*$
shared_nameblock4_conv2/kernel
�
'block4_conv2/kernel/Read/ReadVariableOpReadVariableOpblock4_conv2/kernel*(
_output_shapes
:��*
dtype0
�
block4_conv1/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock4_conv1/bias/*
dtype0*
shape:�*"
shared_nameblock4_conv1/bias
t
%block4_conv1/bias/Read/ReadVariableOpReadVariableOpblock4_conv1/bias*
_output_shapes	
:�*
dtype0
�
block4_conv1/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock4_conv1/kernel/*
dtype0*
shape:��*$
shared_nameblock4_conv1/kernel
�
'block4_conv1/kernel/Read/ReadVariableOpReadVariableOpblock4_conv1/kernel*(
_output_shapes
:��*
dtype0
�
block3_conv3/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock3_conv3/bias/*
dtype0*
shape:�*"
shared_nameblock3_conv3/bias
t
%block3_conv3/bias/Read/ReadVariableOpReadVariableOpblock3_conv3/bias*
_output_shapes	
:�*
dtype0
�
block3_conv3/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock3_conv3/kernel/*
dtype0*
shape:��*$
shared_nameblock3_conv3/kernel
�
'block3_conv3/kernel/Read/ReadVariableOpReadVariableOpblock3_conv3/kernel*(
_output_shapes
:��*
dtype0
�
block3_conv2/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock3_conv2/bias/*
dtype0*
shape:�*"
shared_nameblock3_conv2/bias
t
%block3_conv2/bias/Read/ReadVariableOpReadVariableOpblock3_conv2/bias*
_output_shapes	
:�*
dtype0
�
block3_conv2/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock3_conv2/kernel/*
dtype0*
shape:��*$
shared_nameblock3_conv2/kernel
�
'block3_conv2/kernel/Read/ReadVariableOpReadVariableOpblock3_conv2/kernel*(
_output_shapes
:��*
dtype0
�
block3_conv1/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock3_conv1/bias/*
dtype0*
shape:�*"
shared_nameblock3_conv1/bias
t
%block3_conv1/bias/Read/ReadVariableOpReadVariableOpblock3_conv1/bias*
_output_shapes	
:�*
dtype0
�
block3_conv1/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock3_conv1/kernel/*
dtype0*
shape:��*$
shared_nameblock3_conv1/kernel
�
'block3_conv1/kernel/Read/ReadVariableOpReadVariableOpblock3_conv1/kernel*(
_output_shapes
:��*
dtype0
�
block2_conv2/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock2_conv2/bias/*
dtype0*
shape:�*"
shared_nameblock2_conv2/bias
t
%block2_conv2/bias/Read/ReadVariableOpReadVariableOpblock2_conv2/bias*
_output_shapes	
:�*
dtype0
�
block2_conv2/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock2_conv2/kernel/*
dtype0*
shape:��*$
shared_nameblock2_conv2/kernel
�
'block2_conv2/kernel/Read/ReadVariableOpReadVariableOpblock2_conv2/kernel*(
_output_shapes
:��*
dtype0
�
block2_conv1/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock2_conv1/bias/*
dtype0*
shape:�*"
shared_nameblock2_conv1/bias
t
%block2_conv1/bias/Read/ReadVariableOpReadVariableOpblock2_conv1/bias*
_output_shapes	
:�*
dtype0
�
block2_conv1/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock2_conv1/kernel/*
dtype0*
shape:@�*$
shared_nameblock2_conv1/kernel
�
'block2_conv1/kernel/Read/ReadVariableOpReadVariableOpblock2_conv1/kernel*'
_output_shapes
:@�*
dtype0
�
block1_conv2/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock1_conv2/bias/*
dtype0*
shape:@*"
shared_nameblock1_conv2/bias
s
%block1_conv2/bias/Read/ReadVariableOpReadVariableOpblock1_conv2/bias*
_output_shapes
:@*
dtype0
�
block1_conv2/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock1_conv2/kernel/*
dtype0*
shape:@@*$
shared_nameblock1_conv2/kernel
�
'block1_conv2/kernel/Read/ReadVariableOpReadVariableOpblock1_conv2/kernel*&
_output_shapes
:@@*
dtype0
�
block1_conv1/biasVarHandleOp*
_output_shapes
: *"

debug_nameblock1_conv1/bias/*
dtype0*
shape:@*"
shared_nameblock1_conv1/bias
s
%block1_conv1/bias/Read/ReadVariableOpReadVariableOpblock1_conv1/bias*
_output_shapes
:@*
dtype0
�
block1_conv1/kernelVarHandleOp*
_output_shapes
: *$

debug_nameblock1_conv1/kernel/*
dtype0*
shape:@*$
shared_nameblock1_conv1/kernel
�
'block1_conv1/kernel/Read/ReadVariableOpReadVariableOpblock1_conv1/kernel*&
_output_shapes
:@*
dtype0
�

final/biasVarHandleOp*
_output_shapes
: *

debug_namefinal/bias/*
dtype0*
shape:*
shared_name
final/bias
e
final/bias/Read/ReadVariableOpReadVariableOp
final/bias*
_output_shapes
:*
dtype0
�
final/kernelVarHandleOp*
_output_shapes
: *

debug_namefinal/kernel/*
dtype0*
shape:	�*
shared_namefinal/kernel
n
 final/kernel/Read/ReadVariableOpReadVariableOpfinal/kernel*
_output_shapes
:	�*
dtype0
�
dense1/biasVarHandleOp*
_output_shapes
: *

debug_namedense1/bias/*
dtype0*
shape:�*
shared_namedense1/bias
h
dense1/bias/Read/ReadVariableOpReadVariableOpdense1/bias*
_output_shapes	
:�*
dtype0
�
dense1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense1/kernel/*
dtype0*
shape:���*
shared_namedense1/kernel
r
!dense1/kernel/Read/ReadVariableOpReadVariableOpdense1/kernel*!
_output_shapes
:���*
dtype0
�
serving_default_vgg16_inputPlaceholder*1
_output_shapes
:�����������*
dtype0*&
shape:�����������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_vgg16_inputblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/biasdense1/kerneldense1/biasfinal/kernel
final/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *+
f&R$
"__inference_signature_wrapper_2067

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
 layer_with_weights-12
 layer-17
!layer-18
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses* 
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator* 
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias*
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias*
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
W18
X19
Y20
Z21
[22
\23
]24
^25
;26
<27
C28
D29*
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
W18
X19
Y20
Z21
[22
\23
]24
^25
;26
<27
C28
D29*
* 
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

dtrace_0
etrace_1* 

ftrace_0
gtrace_1* 
* 
�
h
_variables
i_iterations
j_learning_rate
k_index_dict
l
_momentums
m_velocities
n_update_step_xla*

oserving_default* 
* 
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

Ekernel
Fbias
 v_jit_compiled_convolution_op*
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

Gkernel
Hbias
 }_jit_compiled_convolution_op*
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ikernel
Jbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Kkernel
Lbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Mkernel
Nbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Okernel
Pbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Qkernel
Rbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Skernel
Tbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ukernel
Vbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Wkernel
Xbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ykernel
Zbias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

[kernel
\bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

]kernel
^bias
!�_jit_compiled_convolution_op*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
W18
X19
Y20
Z21
[22
\23
]24
^25*
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
W18
X19
Y20
Z21
[22
\23
]24
^25*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses* 

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

;0
<1*

;0
<1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
]W
VARIABLE_VALUEdense1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEdense1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

C0
D1*

C0
D1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEfinal/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
final/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock1_conv2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock1_conv2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock2_conv2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock2_conv2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEblock3_conv1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEblock3_conv1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock3_conv3/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock3_conv3/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock4_conv3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock4_conv3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv1/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv1/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv2/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv2/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
TN
VARIABLE_VALUEblock5_conv3/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEblock5_conv3/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

�0
�1*
* 
* 
* 
* 
* 
* 
�
i0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29*
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29*
* 
* 

E0
F1*

E0
F1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

G0
H1*

G0
H1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

I0
J1*

I0
J1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

K0
L1*

K0
L1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

M0
N1*

M0
N1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

O0
P1*

O0
P1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

Q0
R1*

Q0
R1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

S0
T1*

S0
T1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

U0
V1*

U0
V1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

W0
X1*

W0
X1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

Y0
Z1*

Y0
Z1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

[0
\1*

[0
\1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

]0
^1*

]0
^1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
�	variables
�	keras_api

�total

�count*
M
�	variables
�	keras_api

�total

�count
�
_fn_kwargs*
`Z
VARIABLE_VALUEm/block1_conv1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/block1_conv1/kernel1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEm/block1_conv1/bias1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEv/block1_conv1/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/block1_conv2/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEv/block1_conv2/kernel1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEm/block1_conv2/bias1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEv/block1_conv2/bias1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEm/block2_conv1/kernel1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/block2_conv1/kernel2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEm/block2_conv1/bias2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEv/block2_conv1/bias2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/block2_conv2/kernel2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/block2_conv2/kernel2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEm/block2_conv2/bias2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEv/block2_conv2/bias2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/block3_conv1/kernel2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/block3_conv1/kernel2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEm/block3_conv1/bias2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEv/block3_conv1/bias2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/block3_conv2/kernel2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/block3_conv2/kernel2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEm/block3_conv2/bias2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEv/block3_conv2/bias2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/block3_conv3/kernel2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/block3_conv3/kernel2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEm/block3_conv3/bias2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEv/block3_conv3/bias2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/block4_conv1/kernel2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/block4_conv1/kernel2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEm/block4_conv1/bias2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEv/block4_conv1/bias2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/block4_conv2/kernel2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/block4_conv2/kernel2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEm/block4_conv2/bias2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEv/block4_conv2/bias2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/block4_conv3/kernel2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/block4_conv3/kernel2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEm/block4_conv3/bias2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEv/block4_conv3/bias2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/block5_conv1/kernel2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/block5_conv1/kernel2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEm/block5_conv1/bias2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEv/block5_conv1/bias2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/block5_conv2/kernel2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/block5_conv2/kernel2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEm/block5_conv2/bias2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEv/block5_conv2/bias2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEm/block5_conv3/kernel2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEv/block5_conv3/kernel2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEm/block5_conv3/bias2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEv/block5_conv3/bias2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEm/dense1/kernel2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEv/dense1/kernel2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEm/dense1/bias2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEv/dense1/bias2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEm/final/kernel2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEv/final/kernel2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEm/final/bias2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEv/final/bias2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

�0
�1*

�	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense1/kerneldense1/biasfinal/kernel
final/biasblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/bias	iterationlearning_ratem/block1_conv1/kernelv/block1_conv1/kernelm/block1_conv1/biasv/block1_conv1/biasm/block1_conv2/kernelv/block1_conv2/kernelm/block1_conv2/biasv/block1_conv2/biasm/block2_conv1/kernelv/block2_conv1/kernelm/block2_conv1/biasv/block2_conv1/biasm/block2_conv2/kernelv/block2_conv2/kernelm/block2_conv2/biasv/block2_conv2/biasm/block3_conv1/kernelv/block3_conv1/kernelm/block3_conv1/biasv/block3_conv1/biasm/block3_conv2/kernelv/block3_conv2/kernelm/block3_conv2/biasv/block3_conv2/biasm/block3_conv3/kernelv/block3_conv3/kernelm/block3_conv3/biasv/block3_conv3/biasm/block4_conv1/kernelv/block4_conv1/kernelm/block4_conv1/biasv/block4_conv1/biasm/block4_conv2/kernelv/block4_conv2/kernelm/block4_conv2/biasv/block4_conv2/biasm/block4_conv3/kernelv/block4_conv3/kernelm/block4_conv3/biasv/block4_conv3/biasm/block5_conv1/kernelv/block5_conv1/kernelm/block5_conv1/biasv/block5_conv1/biasm/block5_conv2/kernelv/block5_conv2/kernelm/block5_conv2/biasv/block5_conv2/biasm/block5_conv3/kernelv/block5_conv3/kernelm/block5_conv3/biasv/block5_conv3/biasm/dense1/kernelv/dense1/kernelm/dense1/biasv/dense1/biasm/final/kernelv/final/kernelm/final/biasv/final/biastotal_1count_1totalcountConst*m
Tinf
d2b*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *&
f!R
__inference__traced_save_3053
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense1/kerneldense1/biasfinal/kernel
final/biasblock1_conv1/kernelblock1_conv1/biasblock1_conv2/kernelblock1_conv2/biasblock2_conv1/kernelblock2_conv1/biasblock2_conv2/kernelblock2_conv2/biasblock3_conv1/kernelblock3_conv1/biasblock3_conv2/kernelblock3_conv2/biasblock3_conv3/kernelblock3_conv3/biasblock4_conv1/kernelblock4_conv1/biasblock4_conv2/kernelblock4_conv2/biasblock4_conv3/kernelblock4_conv3/biasblock5_conv1/kernelblock5_conv1/biasblock5_conv2/kernelblock5_conv2/biasblock5_conv3/kernelblock5_conv3/bias	iterationlearning_ratem/block1_conv1/kernelv/block1_conv1/kernelm/block1_conv1/biasv/block1_conv1/biasm/block1_conv2/kernelv/block1_conv2/kernelm/block1_conv2/biasv/block1_conv2/biasm/block2_conv1/kernelv/block2_conv1/kernelm/block2_conv1/biasv/block2_conv1/biasm/block2_conv2/kernelv/block2_conv2/kernelm/block2_conv2/biasv/block2_conv2/biasm/block3_conv1/kernelv/block3_conv1/kernelm/block3_conv1/biasv/block3_conv1/biasm/block3_conv2/kernelv/block3_conv2/kernelm/block3_conv2/biasv/block3_conv2/biasm/block3_conv3/kernelv/block3_conv3/kernelm/block3_conv3/biasv/block3_conv3/biasm/block4_conv1/kernelv/block4_conv1/kernelm/block4_conv1/biasv/block4_conv1/biasm/block4_conv2/kernelv/block4_conv2/kernelm/block4_conv2/biasv/block4_conv2/biasm/block4_conv3/kernelv/block4_conv3/kernelm/block4_conv3/biasv/block4_conv3/biasm/block5_conv1/kernelv/block5_conv1/kernelm/block5_conv1/biasv/block5_conv1/biasm/block5_conv2/kernelv/block5_conv2/kernelm/block5_conv2/biasv/block5_conv2/biasm/block5_conv3/kernelv/block5_conv3/kernelm/block5_conv3/biasv/block5_conv3/biasm/dense1/kernelv/dense1/kernelm/dense1/biasv/dense1/biasm/final/kernelv/final/kernelm/final/biasv/final/biastotal_1count_1totalcount*l
Tine
c2a*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__traced_restore_3350��
�
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_1116

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2235

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������pp�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������pp�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������pp�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:���������pp�
 
_user_specified_nameinputs
�
`
B__inference_dropout1_layer_call_and_return_conditional_losses_2105

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:�����������]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:�����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_block5_conv3_layer_call_and_return_conditional_losses_1340

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
$__inference_vgg16_layer_call_fn_1536
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_1422x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name1532:$ 

_user_specified_name1530:$ 

_user_specified_name1528:$ 

_user_specified_name1526:$ 

_user_specified_name1524:$ 

_user_specified_name1522:$ 

_user_specified_name1520:$ 

_user_specified_name1518:$ 

_user_specified_name1516:$ 

_user_specified_name1514:$ 

_user_specified_name1512:$ 

_user_specified_name1510:$ 

_user_specified_name1508:$ 

_user_specified_name1506:$ 

_user_specified_name1504:$ 

_user_specified_name1502:$
 

_user_specified_name1500:$	 

_user_specified_name1498:$ 

_user_specified_name1496:$ 

_user_specified_name1494:$ 

_user_specified_name1492:$ 

_user_specified_name1490:$ 

_user_specified_name1488:$ 

_user_specified_name1486:$ 

_user_specified_name1484:$ 

_user_specified_name1482:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
F
*__inference_block3_pool_layer_call_fn_2310

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_1106�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_1106

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
F__inference_block3_conv2_layer_call_and_return_conditional_losses_1226

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2405

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_block3_conv3_layer_call_fn_2294

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_1242x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������88�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2290:$ 

_user_specified_name2288:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
+__inference_block2_conv2_layer_call_fn_2224

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_1193x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������pp�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������pp�: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2220:$ 

_user_specified_name2218:X T
0
_output_shapes
:���������pp�
 
_user_specified_nameinputs
�
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_2245

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
+__inference_block5_conv2_layer_call_fn_2414

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_1324x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2410:$ 

_user_specified_name2408:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_block5_pool_layer_call_fn_2450

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_1126�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
+__inference_block4_conv1_layer_call_fn_2324

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_1259x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2320:$ 

_user_specified_name2318:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_block2_pool_layer_call_fn_2240

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_1096�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
F__inference_block3_conv1_layer_call_and_return_conditional_losses_1210

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
+__inference_block4_conv2_layer_call_fn_2344

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_1275x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2340:$ 

_user_specified_name2338:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_block3_conv2_layer_call_fn_2274

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_1226x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������88�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2270:$ 

_user_specified_name2268:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
a
E__inference_block4_pool_layer_call_and_return_conditional_losses_2385

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
F__inference_block4_conv2_layer_call_and_return_conditional_losses_1275

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_block1_conv1_layer_call_and_return_conditional_losses_1144

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�

�
?__inference_final_layer_call_and_return_conditional_losses_2145

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_block4_conv3_layer_call_and_return_conditional_losses_1291

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_block3_conv3_layer_call_and_return_conditional_losses_1242

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
+__inference_block1_conv1_layer_call_fn_2154

inputs!
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_1144y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2150:$ 

_user_specified_name2148:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2265

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
F__inference_block5_conv3_layer_call_and_return_conditional_losses_2445

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_block4_conv3_layer_call_fn_2364

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_1291x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2360:$ 

_user_specified_name2358:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_block5_conv3_layer_call_fn_2434

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_1340x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2430:$ 

_user_specified_name2428:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_block4_conv1_layer_call_and_return_conditional_losses_1259

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
ɲ
�=
 __inference__traced_restore_3350
file_prefix3
assignvariableop_dense1_kernel:���-
assignvariableop_1_dense1_bias:	�2
assignvariableop_2_final_kernel:	�+
assignvariableop_3_final_bias:@
&assignvariableop_4_block1_conv1_kernel:@2
$assignvariableop_5_block1_conv1_bias:@@
&assignvariableop_6_block1_conv2_kernel:@@2
$assignvariableop_7_block1_conv2_bias:@A
&assignvariableop_8_block2_conv1_kernel:@�3
$assignvariableop_9_block2_conv1_bias:	�C
'assignvariableop_10_block2_conv2_kernel:��4
%assignvariableop_11_block2_conv2_bias:	�C
'assignvariableop_12_block3_conv1_kernel:��4
%assignvariableop_13_block3_conv1_bias:	�C
'assignvariableop_14_block3_conv2_kernel:��4
%assignvariableop_15_block3_conv2_bias:	�C
'assignvariableop_16_block3_conv3_kernel:��4
%assignvariableop_17_block3_conv3_bias:	�C
'assignvariableop_18_block4_conv1_kernel:��4
%assignvariableop_19_block4_conv1_bias:	�C
'assignvariableop_20_block4_conv2_kernel:��4
%assignvariableop_21_block4_conv2_bias:	�C
'assignvariableop_22_block4_conv3_kernel:��4
%assignvariableop_23_block4_conv3_bias:	�C
'assignvariableop_24_block5_conv1_kernel:��4
%assignvariableop_25_block5_conv1_bias:	�C
'assignvariableop_26_block5_conv2_kernel:��4
%assignvariableop_27_block5_conv2_bias:	�C
'assignvariableop_28_block5_conv3_kernel:��4
%assignvariableop_29_block5_conv3_bias:	�'
assignvariableop_30_iteration:	 +
!assignvariableop_31_learning_rate: C
)assignvariableop_32_m_block1_conv1_kernel:@C
)assignvariableop_33_v_block1_conv1_kernel:@5
'assignvariableop_34_m_block1_conv1_bias:@5
'assignvariableop_35_v_block1_conv1_bias:@C
)assignvariableop_36_m_block1_conv2_kernel:@@C
)assignvariableop_37_v_block1_conv2_kernel:@@5
'assignvariableop_38_m_block1_conv2_bias:@5
'assignvariableop_39_v_block1_conv2_bias:@D
)assignvariableop_40_m_block2_conv1_kernel:@�D
)assignvariableop_41_v_block2_conv1_kernel:@�6
'assignvariableop_42_m_block2_conv1_bias:	�6
'assignvariableop_43_v_block2_conv1_bias:	�E
)assignvariableop_44_m_block2_conv2_kernel:��E
)assignvariableop_45_v_block2_conv2_kernel:��6
'assignvariableop_46_m_block2_conv2_bias:	�6
'assignvariableop_47_v_block2_conv2_bias:	�E
)assignvariableop_48_m_block3_conv1_kernel:��E
)assignvariableop_49_v_block3_conv1_kernel:��6
'assignvariableop_50_m_block3_conv1_bias:	�6
'assignvariableop_51_v_block3_conv1_bias:	�E
)assignvariableop_52_m_block3_conv2_kernel:��E
)assignvariableop_53_v_block3_conv2_kernel:��6
'assignvariableop_54_m_block3_conv2_bias:	�6
'assignvariableop_55_v_block3_conv2_bias:	�E
)assignvariableop_56_m_block3_conv3_kernel:��E
)assignvariableop_57_v_block3_conv3_kernel:��6
'assignvariableop_58_m_block3_conv3_bias:	�6
'assignvariableop_59_v_block3_conv3_bias:	�E
)assignvariableop_60_m_block4_conv1_kernel:��E
)assignvariableop_61_v_block4_conv1_kernel:��6
'assignvariableop_62_m_block4_conv1_bias:	�6
'assignvariableop_63_v_block4_conv1_bias:	�E
)assignvariableop_64_m_block4_conv2_kernel:��E
)assignvariableop_65_v_block4_conv2_kernel:��6
'assignvariableop_66_m_block4_conv2_bias:	�6
'assignvariableop_67_v_block4_conv2_bias:	�E
)assignvariableop_68_m_block4_conv3_kernel:��E
)assignvariableop_69_v_block4_conv3_kernel:��6
'assignvariableop_70_m_block4_conv3_bias:	�6
'assignvariableop_71_v_block4_conv3_bias:	�E
)assignvariableop_72_m_block5_conv1_kernel:��E
)assignvariableop_73_v_block5_conv1_kernel:��6
'assignvariableop_74_m_block5_conv1_bias:	�6
'assignvariableop_75_v_block5_conv1_bias:	�E
)assignvariableop_76_m_block5_conv2_kernel:��E
)assignvariableop_77_v_block5_conv2_kernel:��6
'assignvariableop_78_m_block5_conv2_bias:	�6
'assignvariableop_79_v_block5_conv2_bias:	�E
)assignvariableop_80_m_block5_conv3_kernel:��E
)assignvariableop_81_v_block5_conv3_kernel:��6
'assignvariableop_82_m_block5_conv3_bias:	�6
'assignvariableop_83_v_block5_conv3_bias:	�8
#assignvariableop_84_m_dense1_kernel:���8
#assignvariableop_85_v_dense1_kernel:���0
!assignvariableop_86_m_dense1_bias:	�0
!assignvariableop_87_v_dense1_bias:	�5
"assignvariableop_88_m_final_kernel:	�5
"assignvariableop_89_v_final_kernel:	�.
 assignvariableop_90_m_final_bias:.
 assignvariableop_91_v_final_bias:%
assignvariableop_92_total_1: %
assignvariableop_93_count_1: #
assignvariableop_94_total: #
assignvariableop_95_count: 
identity_97��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_64�AssignVariableOp_65�AssignVariableOp_66�AssignVariableOp_67�AssignVariableOp_68�AssignVariableOp_69�AssignVariableOp_7�AssignVariableOp_70�AssignVariableOp_71�AssignVariableOp_72�AssignVariableOp_73�AssignVariableOp_74�AssignVariableOp_75�AssignVariableOp_76�AssignVariableOp_77�AssignVariableOp_78�AssignVariableOp_79�AssignVariableOp_8�AssignVariableOp_80�AssignVariableOp_81�AssignVariableOp_82�AssignVariableOp_83�AssignVariableOp_84�AssignVariableOp_85�AssignVariableOp_86�AssignVariableOp_87�AssignVariableOp_88�AssignVariableOp_89�AssignVariableOp_9�AssignVariableOp_90�AssignVariableOp_91�AssignVariableOp_92�AssignVariableOp_93�AssignVariableOp_94�AssignVariableOp_95�%
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*�%
value�%B�%aB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*�
value�B�aB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*o
dtypese
c2a	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOpassignvariableop_dense1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpassignvariableop_2_final_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOpassignvariableop_3_final_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp&assignvariableop_4_block1_conv1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp$assignvariableop_5_block1_conv1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp&assignvariableop_6_block1_conv2_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp$assignvariableop_7_block1_conv2_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_block2_conv1_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp$assignvariableop_9_block2_conv1_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_block2_conv2_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp%assignvariableop_11_block2_conv2_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp'assignvariableop_12_block3_conv1_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp%assignvariableop_13_block3_conv1_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp'assignvariableop_14_block3_conv2_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp%assignvariableop_15_block3_conv2_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp'assignvariableop_16_block3_conv3_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp%assignvariableop_17_block3_conv3_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp'assignvariableop_18_block4_conv1_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp%assignvariableop_19_block4_conv1_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_block4_conv2_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp%assignvariableop_21_block4_conv2_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp'assignvariableop_22_block4_conv3_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOp%assignvariableop_23_block4_conv3_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOp'assignvariableop_24_block5_conv1_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOp%assignvariableop_25_block5_conv1_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOp'assignvariableop_26_block5_conv2_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_27AssignVariableOp%assignvariableop_27_block5_conv2_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_28AssignVariableOp'assignvariableop_28_block5_conv3_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_29AssignVariableOp%assignvariableop_29_block5_conv3_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0	*
_output_shapes
:�
AssignVariableOp_30AssignVariableOpassignvariableop_30_iterationIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_31AssignVariableOp!assignvariableop_31_learning_rateIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_32AssignVariableOp)assignvariableop_32_m_block1_conv1_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_33AssignVariableOp)assignvariableop_33_v_block1_conv1_kernelIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_34AssignVariableOp'assignvariableop_34_m_block1_conv1_biasIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_35AssignVariableOp'assignvariableop_35_v_block1_conv1_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_m_block1_conv2_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_37AssignVariableOp)assignvariableop_37_v_block1_conv2_kernelIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_38AssignVariableOp'assignvariableop_38_m_block1_conv2_biasIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_39AssignVariableOp'assignvariableop_39_v_block1_conv2_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_m_block2_conv1_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_41AssignVariableOp)assignvariableop_41_v_block2_conv1_kernelIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_42AssignVariableOp'assignvariableop_42_m_block2_conv1_biasIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_43AssignVariableOp'assignvariableop_43_v_block2_conv1_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_44AssignVariableOp)assignvariableop_44_m_block2_conv2_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_45AssignVariableOp)assignvariableop_45_v_block2_conv2_kernelIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_46AssignVariableOp'assignvariableop_46_m_block2_conv2_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_47AssignVariableOp'assignvariableop_47_v_block2_conv2_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_48AssignVariableOp)assignvariableop_48_m_block3_conv1_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_49AssignVariableOp)assignvariableop_49_v_block3_conv1_kernelIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_50AssignVariableOp'assignvariableop_50_m_block3_conv1_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_51AssignVariableOp'assignvariableop_51_v_block3_conv1_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_52AssignVariableOp)assignvariableop_52_m_block3_conv2_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_53AssignVariableOp)assignvariableop_53_v_block3_conv2_kernelIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_54AssignVariableOp'assignvariableop_54_m_block3_conv2_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_55AssignVariableOp'assignvariableop_55_v_block3_conv2_biasIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_56AssignVariableOp)assignvariableop_56_m_block3_conv3_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_57AssignVariableOp)assignvariableop_57_v_block3_conv3_kernelIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_58AssignVariableOp'assignvariableop_58_m_block3_conv3_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_59AssignVariableOp'assignvariableop_59_v_block3_conv3_biasIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_60AssignVariableOp)assignvariableop_60_m_block4_conv1_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_61AssignVariableOp)assignvariableop_61_v_block4_conv1_kernelIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_62AssignVariableOp'assignvariableop_62_m_block4_conv1_biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_63AssignVariableOp'assignvariableop_63_v_block4_conv1_biasIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_64AssignVariableOp)assignvariableop_64_m_block4_conv2_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_65AssignVariableOp)assignvariableop_65_v_block4_conv2_kernelIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_66AssignVariableOp'assignvariableop_66_m_block4_conv2_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_67AssignVariableOp'assignvariableop_67_v_block4_conv2_biasIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_68AssignVariableOp)assignvariableop_68_m_block4_conv3_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_69AssignVariableOp)assignvariableop_69_v_block4_conv3_kernelIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_70AssignVariableOp'assignvariableop_70_m_block4_conv3_biasIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_71AssignVariableOp'assignvariableop_71_v_block4_conv3_biasIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_72AssignVariableOp)assignvariableop_72_m_block5_conv1_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_73AssignVariableOp)assignvariableop_73_v_block5_conv1_kernelIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_74AssignVariableOp'assignvariableop_74_m_block5_conv1_biasIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_75AssignVariableOp'assignvariableop_75_v_block5_conv1_biasIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_76AssignVariableOp)assignvariableop_76_m_block5_conv2_kernelIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_77AssignVariableOp)assignvariableop_77_v_block5_conv2_kernelIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_78AssignVariableOp'assignvariableop_78_m_block5_conv2_biasIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_79AssignVariableOp'assignvariableop_79_v_block5_conv2_biasIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_80AssignVariableOp)assignvariableop_80_m_block5_conv3_kernelIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_81AssignVariableOp)assignvariableop_81_v_block5_conv3_kernelIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_82AssignVariableOp'assignvariableop_82_m_block5_conv3_biasIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_83AssignVariableOp'assignvariableop_83_v_block5_conv3_biasIdentity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_84AssignVariableOp#assignvariableop_84_m_dense1_kernelIdentity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_85AssignVariableOp#assignvariableop_85_v_dense1_kernelIdentity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_86AssignVariableOp!assignvariableop_86_m_dense1_biasIdentity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_87AssignVariableOp!assignvariableop_87_v_dense1_biasIdentity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_88AssignVariableOp"assignvariableop_88_m_final_kernelIdentity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_89AssignVariableOp"assignvariableop_89_v_final_kernelIdentity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_90AssignVariableOp assignvariableop_90_m_final_biasIdentity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_91AssignVariableOp assignvariableop_91_v_final_biasIdentity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_92AssignVariableOpassignvariableop_92_total_1Identity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_93AssignVariableOpassignvariableop_93_count_1Identity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_94AssignVariableOpassignvariableop_94_totalIdentity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_95AssignVariableOpassignvariableop_95_countIdentity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_96Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_97IdentityIdentity_96:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95*
_output_shapes
 "#
identity_97Identity_97:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:%`!

_user_specified_namecount:%_!

_user_specified_nametotal:'^#
!
_user_specified_name	count_1:']#
!
_user_specified_name	total_1:,\(
&
_user_specified_namev/final/bias:,[(
&
_user_specified_namem/final/bias:.Z*
(
_user_specified_namev/final/kernel:.Y*
(
_user_specified_namem/final/kernel:-X)
'
_user_specified_namev/dense1/bias:-W)
'
_user_specified_namem/dense1/bias:/V+
)
_user_specified_namev/dense1/kernel:/U+
)
_user_specified_namem/dense1/kernel:3T/
-
_user_specified_namev/block5_conv3/bias:3S/
-
_user_specified_namem/block5_conv3/bias:5R1
/
_user_specified_namev/block5_conv3/kernel:5Q1
/
_user_specified_namem/block5_conv3/kernel:3P/
-
_user_specified_namev/block5_conv2/bias:3O/
-
_user_specified_namem/block5_conv2/bias:5N1
/
_user_specified_namev/block5_conv2/kernel:5M1
/
_user_specified_namem/block5_conv2/kernel:3L/
-
_user_specified_namev/block5_conv1/bias:3K/
-
_user_specified_namem/block5_conv1/bias:5J1
/
_user_specified_namev/block5_conv1/kernel:5I1
/
_user_specified_namem/block5_conv1/kernel:3H/
-
_user_specified_namev/block4_conv3/bias:3G/
-
_user_specified_namem/block4_conv3/bias:5F1
/
_user_specified_namev/block4_conv3/kernel:5E1
/
_user_specified_namem/block4_conv3/kernel:3D/
-
_user_specified_namev/block4_conv2/bias:3C/
-
_user_specified_namem/block4_conv2/bias:5B1
/
_user_specified_namev/block4_conv2/kernel:5A1
/
_user_specified_namem/block4_conv2/kernel:3@/
-
_user_specified_namev/block4_conv1/bias:3?/
-
_user_specified_namem/block4_conv1/bias:5>1
/
_user_specified_namev/block4_conv1/kernel:5=1
/
_user_specified_namem/block4_conv1/kernel:3</
-
_user_specified_namev/block3_conv3/bias:3;/
-
_user_specified_namem/block3_conv3/bias:5:1
/
_user_specified_namev/block3_conv3/kernel:591
/
_user_specified_namem/block3_conv3/kernel:38/
-
_user_specified_namev/block3_conv2/bias:37/
-
_user_specified_namem/block3_conv2/bias:561
/
_user_specified_namev/block3_conv2/kernel:551
/
_user_specified_namem/block3_conv2/kernel:34/
-
_user_specified_namev/block3_conv1/bias:33/
-
_user_specified_namem/block3_conv1/bias:521
/
_user_specified_namev/block3_conv1/kernel:511
/
_user_specified_namem/block3_conv1/kernel:30/
-
_user_specified_namev/block2_conv2/bias:3//
-
_user_specified_namem/block2_conv2/bias:5.1
/
_user_specified_namev/block2_conv2/kernel:5-1
/
_user_specified_namem/block2_conv2/kernel:3,/
-
_user_specified_namev/block2_conv1/bias:3+/
-
_user_specified_namem/block2_conv1/bias:5*1
/
_user_specified_namev/block2_conv1/kernel:5)1
/
_user_specified_namem/block2_conv1/kernel:3(/
-
_user_specified_namev/block1_conv2/bias:3'/
-
_user_specified_namem/block1_conv2/bias:5&1
/
_user_specified_namev/block1_conv2/kernel:5%1
/
_user_specified_namem/block1_conv2/kernel:3$/
-
_user_specified_namev/block1_conv1/bias:3#/
-
_user_specified_namem/block1_conv1/bias:5"1
/
_user_specified_namev/block1_conv1/kernel:5!1
/
_user_specified_namem/block1_conv1/kernel:- )
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:1-
+
_user_specified_nameblock5_conv3/bias:3/
-
_user_specified_nameblock5_conv3/kernel:1-
+
_user_specified_nameblock5_conv2/bias:3/
-
_user_specified_nameblock5_conv2/kernel:1-
+
_user_specified_nameblock5_conv1/bias:3/
-
_user_specified_nameblock5_conv1/kernel:1-
+
_user_specified_nameblock4_conv3/bias:3/
-
_user_specified_nameblock4_conv3/kernel:1-
+
_user_specified_nameblock4_conv2/bias:3/
-
_user_specified_nameblock4_conv2/kernel:1-
+
_user_specified_nameblock4_conv1/bias:3/
-
_user_specified_nameblock4_conv1/kernel:1-
+
_user_specified_nameblock3_conv3/bias:3/
-
_user_specified_nameblock3_conv3/kernel:1-
+
_user_specified_nameblock3_conv2/bias:3/
-
_user_specified_nameblock3_conv2/kernel:1-
+
_user_specified_nameblock3_conv1/bias:3/
-
_user_specified_nameblock3_conv1/kernel:1-
+
_user_specified_nameblock2_conv2/bias:3/
-
_user_specified_nameblock2_conv2/kernel:1
-
+
_user_specified_nameblock2_conv1/bias:3	/
-
_user_specified_nameblock2_conv1/kernel:1-
+
_user_specified_nameblock1_conv2/bias:3/
-
_user_specified_nameblock1_conv2/kernel:1-
+
_user_specified_nameblock1_conv1/bias:3/
-
_user_specified_nameblock1_conv1/kernel:*&
$
_user_specified_name
final/bias:,(
&
_user_specified_namefinal/kernel:+'
%
_user_specified_namedense1/bias:-)
'
_user_specified_namedense1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
F__inference_block2_conv2_layer_call_and_return_conditional_losses_1193

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������pp�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������pp�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������pp�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:���������pp�
 
_user_specified_nameinputs
�
�
F__inference_block1_conv2_layer_call_and_return_conditional_losses_1160

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�

�
?__inference_final_layer_call_and_return_conditional_losses_1755

inputs1
matmul_readvariableop_resource:	�-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:���������`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:���������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_block2_pool_layer_call_and_return_conditional_losses_1096

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_1714

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2215

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������pp�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������pp�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������pp@
 
_user_specified_nameinputs
�

�
@__inference_dense1_layer_call_and_return_conditional_losses_2125

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�

a
B__inference_dropout1_layer_call_and_return_conditional_losses_2100

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:�����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:�����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:�����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*)
_output_shapes
:�����������c
IdentityIdentitydropout/SelectV2:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
"__inference_signature_wrapper_2067
vgg16_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�

unknown_25:���

unknown_26:	�

unknown_27:	�

unknown_28:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvgg16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *(
f#R!
__inference__wrapped_model_1081o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2063:$ 

_user_specified_name2061:$ 

_user_specified_name2059:$ 

_user_specified_name2057:$ 

_user_specified_name2055:$ 

_user_specified_name2053:$ 

_user_specified_name2051:$ 

_user_specified_name2049:$ 

_user_specified_name2047:$ 

_user_specified_name2045:$ 

_user_specified_name2043:$ 

_user_specified_name2041:$ 

_user_specified_name2039:$ 

_user_specified_name2037:$ 

_user_specified_name2035:$ 

_user_specified_name2033:$ 

_user_specified_name2031:$ 

_user_specified_name2029:$ 

_user_specified_name2027:$ 

_user_specified_name2025:$
 

_user_specified_name2023:$	 

_user_specified_name2021:$ 

_user_specified_name2019:$ 

_user_specified_name2017:$ 

_user_specified_name2015:$ 

_user_specified_name2013:$ 

_user_specified_name2011:$ 

_user_specified_name2009:$ 

_user_specified_name2007:$ 

_user_specified_name2005:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg16_input
�
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_1086

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
$__inference_final_layer_call_fn_2134

inputs
unknown:	�
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_final_layer_call_and_return_conditional_losses_1755o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2130:$ 

_user_specified_name2128:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_block1_pool_layer_call_and_return_conditional_losses_2195

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
C
'__inference_dropout1_layer_call_fn_2088

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout1_layer_call_and_return_conditional_losses_1823b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
+__inference_block2_conv1_layer_call_fn_2204

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_1177x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������pp�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp@: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2200:$ 

_user_specified_name2198:W S
/
_output_shapes
:���������pp@
 
_user_specified_nameinputs
�
�
F__inference_block5_conv2_layer_call_and_return_conditional_losses_1324

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
��
� 
__inference__wrapped_model_1081
vgg16_inputV
<vgg16_like_vgg16_block1_conv1_conv2d_readvariableop_resource:@K
=vgg16_like_vgg16_block1_conv1_biasadd_readvariableop_resource:@V
<vgg16_like_vgg16_block1_conv2_conv2d_readvariableop_resource:@@K
=vgg16_like_vgg16_block1_conv2_biasadd_readvariableop_resource:@W
<vgg16_like_vgg16_block2_conv1_conv2d_readvariableop_resource:@�L
=vgg16_like_vgg16_block2_conv1_biasadd_readvariableop_resource:	�X
<vgg16_like_vgg16_block2_conv2_conv2d_readvariableop_resource:��L
=vgg16_like_vgg16_block2_conv2_biasadd_readvariableop_resource:	�X
<vgg16_like_vgg16_block3_conv1_conv2d_readvariableop_resource:��L
=vgg16_like_vgg16_block3_conv1_biasadd_readvariableop_resource:	�X
<vgg16_like_vgg16_block3_conv2_conv2d_readvariableop_resource:��L
=vgg16_like_vgg16_block3_conv2_biasadd_readvariableop_resource:	�X
<vgg16_like_vgg16_block3_conv3_conv2d_readvariableop_resource:��L
=vgg16_like_vgg16_block3_conv3_biasadd_readvariableop_resource:	�X
<vgg16_like_vgg16_block4_conv1_conv2d_readvariableop_resource:��L
=vgg16_like_vgg16_block4_conv1_biasadd_readvariableop_resource:	�X
<vgg16_like_vgg16_block4_conv2_conv2d_readvariableop_resource:��L
=vgg16_like_vgg16_block4_conv2_biasadd_readvariableop_resource:	�X
<vgg16_like_vgg16_block4_conv3_conv2d_readvariableop_resource:��L
=vgg16_like_vgg16_block4_conv3_biasadd_readvariableop_resource:	�X
<vgg16_like_vgg16_block5_conv1_conv2d_readvariableop_resource:��L
=vgg16_like_vgg16_block5_conv1_biasadd_readvariableop_resource:	�X
<vgg16_like_vgg16_block5_conv2_conv2d_readvariableop_resource:��L
=vgg16_like_vgg16_block5_conv2_biasadd_readvariableop_resource:	�X
<vgg16_like_vgg16_block5_conv3_conv2d_readvariableop_resource:��L
=vgg16_like_vgg16_block5_conv3_biasadd_readvariableop_resource:	�E
0vgg16_like_dense1_matmul_readvariableop_resource:���@
1vgg16_like_dense1_biasadd_readvariableop_resource:	�B
/vgg16_like_final_matmul_readvariableop_resource:	�>
0vgg16_like_final_biasadd_readvariableop_resource:
identity��(VGG16_like/dense1/BiasAdd/ReadVariableOp�'VGG16_like/dense1/MatMul/ReadVariableOp�'VGG16_like/final/BiasAdd/ReadVariableOp�&VGG16_like/final/MatMul/ReadVariableOp�4VGG16_like/vgg16/block1_conv1/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block1_conv1/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block1_conv2/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block1_conv2/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block2_conv1/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block2_conv1/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block2_conv2/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block2_conv2/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block3_conv1/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block3_conv1/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block3_conv2/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block3_conv2/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block3_conv3/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block3_conv3/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block4_conv1/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block4_conv1/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block4_conv2/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block4_conv2/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block4_conv3/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block4_conv3/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block5_conv1/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block5_conv1/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block5_conv2/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block5_conv2/Conv2D/ReadVariableOp�4VGG16_like/vgg16/block5_conv3/BiasAdd/ReadVariableOp�3VGG16_like/vgg16/block5_conv3/Conv2D/ReadVariableOp�
3VGG16_like/vgg16/block1_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
$VGG16_like/vgg16/block1_conv1/Conv2DConv2Dvgg16_input;VGG16_like/vgg16/block1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
4VGG16_like/vgg16/block1_conv1/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block1_conv1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
%VGG16_like/vgg16/block1_conv1/BiasAddBiasAdd-VGG16_like/vgg16/block1_conv1/Conv2D:output:0<VGG16_like/vgg16/block1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
"VGG16_like/vgg16/block1_conv1/ReluRelu.VGG16_like/vgg16/block1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
3VGG16_like/vgg16/block1_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block1_conv2_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
$VGG16_like/vgg16/block1_conv2/Conv2DConv2D0VGG16_like/vgg16/block1_conv1/Relu:activations:0;VGG16_like/vgg16/block1_conv2/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
�
4VGG16_like/vgg16/block1_conv2/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block1_conv2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
%VGG16_like/vgg16/block1_conv2/BiasAddBiasAdd-VGG16_like/vgg16/block1_conv2/Conv2D:output:0<VGG16_like/vgg16/block1_conv2/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@�
"VGG16_like/vgg16/block1_conv2/ReluRelu.VGG16_like/vgg16/block1_conv2/BiasAdd:output:0*
T0*1
_output_shapes
:�����������@�
$VGG16_like/vgg16/block1_pool/MaxPoolMaxPool0VGG16_like/vgg16/block1_conv2/Relu:activations:0*/
_output_shapes
:���������pp@*
ksize
*
paddingVALID*
strides
�
3VGG16_like/vgg16/block2_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block2_conv1_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
$VGG16_like/vgg16/block2_conv1/Conv2DConv2D-VGG16_like/vgg16/block1_pool/MaxPool:output:0;VGG16_like/vgg16/block2_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
�
4VGG16_like/vgg16/block2_conv1/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block2_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%VGG16_like/vgg16/block2_conv1/BiasAddBiasAdd-VGG16_like/vgg16/block2_conv1/Conv2D:output:0<VGG16_like/vgg16/block2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp��
"VGG16_like/vgg16/block2_conv1/ReluRelu.VGG16_like/vgg16/block2_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp��
3VGG16_like/vgg16/block2_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block2_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$VGG16_like/vgg16/block2_conv2/Conv2DConv2D0VGG16_like/vgg16/block2_conv1/Relu:activations:0;VGG16_like/vgg16/block2_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
�
4VGG16_like/vgg16/block2_conv2/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block2_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%VGG16_like/vgg16/block2_conv2/BiasAddBiasAdd-VGG16_like/vgg16/block2_conv2/Conv2D:output:0<VGG16_like/vgg16/block2_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp��
"VGG16_like/vgg16/block2_conv2/ReluRelu.VGG16_like/vgg16/block2_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������pp��
$VGG16_like/vgg16/block2_pool/MaxPoolMaxPool0VGG16_like/vgg16/block2_conv2/Relu:activations:0*0
_output_shapes
:���������88�*
ksize
*
paddingVALID*
strides
�
3VGG16_like/vgg16/block3_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block3_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$VGG16_like/vgg16/block3_conv1/Conv2DConv2D-VGG16_like/vgg16/block2_pool/MaxPool:output:0;VGG16_like/vgg16/block3_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
�
4VGG16_like/vgg16/block3_conv1/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block3_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%VGG16_like/vgg16/block3_conv1/BiasAddBiasAdd-VGG16_like/vgg16/block3_conv1/Conv2D:output:0<VGG16_like/vgg16/block3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88��
"VGG16_like/vgg16/block3_conv1/ReluRelu.VGG16_like/vgg16/block3_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:���������88��
3VGG16_like/vgg16/block3_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block3_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$VGG16_like/vgg16/block3_conv2/Conv2DConv2D0VGG16_like/vgg16/block3_conv1/Relu:activations:0;VGG16_like/vgg16/block3_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
�
4VGG16_like/vgg16/block3_conv2/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block3_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%VGG16_like/vgg16/block3_conv2/BiasAddBiasAdd-VGG16_like/vgg16/block3_conv2/Conv2D:output:0<VGG16_like/vgg16/block3_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88��
"VGG16_like/vgg16/block3_conv2/ReluRelu.VGG16_like/vgg16/block3_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:���������88��
3VGG16_like/vgg16/block3_conv3/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block3_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$VGG16_like/vgg16/block3_conv3/Conv2DConv2D0VGG16_like/vgg16/block3_conv2/Relu:activations:0;VGG16_like/vgg16/block3_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
�
4VGG16_like/vgg16/block3_conv3/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block3_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%VGG16_like/vgg16/block3_conv3/BiasAddBiasAdd-VGG16_like/vgg16/block3_conv3/Conv2D:output:0<VGG16_like/vgg16/block3_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88��
"VGG16_like/vgg16/block3_conv3/ReluRelu.VGG16_like/vgg16/block3_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:���������88��
$VGG16_like/vgg16/block3_pool/MaxPoolMaxPool0VGG16_like/vgg16/block3_conv3/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
3VGG16_like/vgg16/block4_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block4_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$VGG16_like/vgg16/block4_conv1/Conv2DConv2D-VGG16_like/vgg16/block3_pool/MaxPool:output:0;VGG16_like/vgg16/block4_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4VGG16_like/vgg16/block4_conv1/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block4_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%VGG16_like/vgg16/block4_conv1/BiasAddBiasAdd-VGG16_like/vgg16/block4_conv1/Conv2D:output:0<VGG16_like/vgg16/block4_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
"VGG16_like/vgg16/block4_conv1/ReluRelu.VGG16_like/vgg16/block4_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
3VGG16_like/vgg16/block4_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block4_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$VGG16_like/vgg16/block4_conv2/Conv2DConv2D0VGG16_like/vgg16/block4_conv1/Relu:activations:0;VGG16_like/vgg16/block4_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4VGG16_like/vgg16/block4_conv2/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block4_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%VGG16_like/vgg16/block4_conv2/BiasAddBiasAdd-VGG16_like/vgg16/block4_conv2/Conv2D:output:0<VGG16_like/vgg16/block4_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
"VGG16_like/vgg16/block4_conv2/ReluRelu.VGG16_like/vgg16/block4_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
3VGG16_like/vgg16/block4_conv3/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block4_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$VGG16_like/vgg16/block4_conv3/Conv2DConv2D0VGG16_like/vgg16/block4_conv2/Relu:activations:0;VGG16_like/vgg16/block4_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4VGG16_like/vgg16/block4_conv3/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block4_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%VGG16_like/vgg16/block4_conv3/BiasAddBiasAdd-VGG16_like/vgg16/block4_conv3/Conv2D:output:0<VGG16_like/vgg16/block4_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
"VGG16_like/vgg16/block4_conv3/ReluRelu.VGG16_like/vgg16/block4_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
$VGG16_like/vgg16/block4_pool/MaxPoolMaxPool0VGG16_like/vgg16/block4_conv3/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
�
3VGG16_like/vgg16/block5_conv1/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block5_conv1_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$VGG16_like/vgg16/block5_conv1/Conv2DConv2D-VGG16_like/vgg16/block4_pool/MaxPool:output:0;VGG16_like/vgg16/block5_conv1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4VGG16_like/vgg16/block5_conv1/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block5_conv1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%VGG16_like/vgg16/block5_conv1/BiasAddBiasAdd-VGG16_like/vgg16/block5_conv1/Conv2D:output:0<VGG16_like/vgg16/block5_conv1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
"VGG16_like/vgg16/block5_conv1/ReluRelu.VGG16_like/vgg16/block5_conv1/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
3VGG16_like/vgg16/block5_conv2/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block5_conv2_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$VGG16_like/vgg16/block5_conv2/Conv2DConv2D0VGG16_like/vgg16/block5_conv1/Relu:activations:0;VGG16_like/vgg16/block5_conv2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4VGG16_like/vgg16/block5_conv2/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block5_conv2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%VGG16_like/vgg16/block5_conv2/BiasAddBiasAdd-VGG16_like/vgg16/block5_conv2/Conv2D:output:0<VGG16_like/vgg16/block5_conv2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
"VGG16_like/vgg16/block5_conv2/ReluRelu.VGG16_like/vgg16/block5_conv2/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
3VGG16_like/vgg16/block5_conv3/Conv2D/ReadVariableOpReadVariableOp<vgg16_like_vgg16_block5_conv3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
$VGG16_like/vgg16/block5_conv3/Conv2DConv2D0VGG16_like/vgg16/block5_conv2/Relu:activations:0;VGG16_like/vgg16/block5_conv3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
�
4VGG16_like/vgg16/block5_conv3/BiasAdd/ReadVariableOpReadVariableOp=vgg16_like_vgg16_block5_conv3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
%VGG16_like/vgg16/block5_conv3/BiasAddBiasAdd-VGG16_like/vgg16/block5_conv3/Conv2D:output:0<VGG16_like/vgg16/block5_conv3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:�����������
"VGG16_like/vgg16/block5_conv3/ReluRelu.VGG16_like/vgg16/block5_conv3/BiasAdd:output:0*
T0*0
_output_shapes
:�����������
$VGG16_like/vgg16/block5_pool/MaxPoolMaxPool0VGG16_like/vgg16/block5_conv3/Relu:activations:0*0
_output_shapes
:����������*
ksize
*
paddingVALID*
strides
i
VGG16_like/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  �
VGG16_like/flatten/ReshapeReshape-VGG16_like/vgg16/block5_pool/MaxPool:output:0!VGG16_like/flatten/Const:output:0*
T0*)
_output_shapes
:������������
VGG16_like/dropout1/IdentityIdentity#VGG16_like/flatten/Reshape:output:0*
T0*)
_output_shapes
:������������
'VGG16_like/dense1/MatMul/ReadVariableOpReadVariableOp0vgg16_like_dense1_matmul_readvariableop_resource*!
_output_shapes
:���*
dtype0�
VGG16_like/dense1/MatMulMatMul%VGG16_like/dropout1/Identity:output:0/VGG16_like/dense1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
(VGG16_like/dense1/BiasAdd/ReadVariableOpReadVariableOp1vgg16_like_dense1_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
VGG16_like/dense1/BiasAddBiasAdd"VGG16_like/dense1/MatMul:product:00VGG16_like/dense1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������u
VGG16_like/dense1/ReluRelu"VGG16_like/dense1/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
&VGG16_like/final/MatMul/ReadVariableOpReadVariableOp/vgg16_like_final_matmul_readvariableop_resource*
_output_shapes
:	�*
dtype0�
VGG16_like/final/MatMulMatMul$VGG16_like/dense1/Relu:activations:0.VGG16_like/final/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
'VGG16_like/final/BiasAdd/ReadVariableOpReadVariableOp0vgg16_like_final_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
VGG16_like/final/BiasAddBiasAdd!VGG16_like/final/MatMul:product:0/VGG16_like/final/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������x
VGG16_like/final/SoftmaxSoftmax!VGG16_like/final/BiasAdd:output:0*
T0*'
_output_shapes
:���������q
IdentityIdentity"VGG16_like/final/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp)^VGG16_like/dense1/BiasAdd/ReadVariableOp(^VGG16_like/dense1/MatMul/ReadVariableOp(^VGG16_like/final/BiasAdd/ReadVariableOp'^VGG16_like/final/MatMul/ReadVariableOp5^VGG16_like/vgg16/block1_conv1/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block1_conv1/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block1_conv2/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block1_conv2/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block2_conv1/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block2_conv1/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block2_conv2/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block2_conv2/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block3_conv1/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block3_conv1/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block3_conv2/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block3_conv2/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block3_conv3/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block3_conv3/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block4_conv1/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block4_conv1/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block4_conv2/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block4_conv2/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block4_conv3/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block4_conv3/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block5_conv1/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block5_conv1/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block5_conv2/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block5_conv2/Conv2D/ReadVariableOp5^VGG16_like/vgg16/block5_conv3/BiasAdd/ReadVariableOp4^VGG16_like/vgg16/block5_conv3/Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2T
(VGG16_like/dense1/BiasAdd/ReadVariableOp(VGG16_like/dense1/BiasAdd/ReadVariableOp2R
'VGG16_like/dense1/MatMul/ReadVariableOp'VGG16_like/dense1/MatMul/ReadVariableOp2R
'VGG16_like/final/BiasAdd/ReadVariableOp'VGG16_like/final/BiasAdd/ReadVariableOp2P
&VGG16_like/final/MatMul/ReadVariableOp&VGG16_like/final/MatMul/ReadVariableOp2l
4VGG16_like/vgg16/block1_conv1/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block1_conv1/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block1_conv1/Conv2D/ReadVariableOp3VGG16_like/vgg16/block1_conv1/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block1_conv2/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block1_conv2/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block1_conv2/Conv2D/ReadVariableOp3VGG16_like/vgg16/block1_conv2/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block2_conv1/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block2_conv1/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block2_conv1/Conv2D/ReadVariableOp3VGG16_like/vgg16/block2_conv1/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block2_conv2/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block2_conv2/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block2_conv2/Conv2D/ReadVariableOp3VGG16_like/vgg16/block2_conv2/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block3_conv1/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block3_conv1/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block3_conv1/Conv2D/ReadVariableOp3VGG16_like/vgg16/block3_conv1/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block3_conv2/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block3_conv2/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block3_conv2/Conv2D/ReadVariableOp3VGG16_like/vgg16/block3_conv2/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block3_conv3/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block3_conv3/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block3_conv3/Conv2D/ReadVariableOp3VGG16_like/vgg16/block3_conv3/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block4_conv1/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block4_conv1/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block4_conv1/Conv2D/ReadVariableOp3VGG16_like/vgg16/block4_conv1/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block4_conv2/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block4_conv2/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block4_conv2/Conv2D/ReadVariableOp3VGG16_like/vgg16/block4_conv2/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block4_conv3/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block4_conv3/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block4_conv3/Conv2D/ReadVariableOp3VGG16_like/vgg16/block4_conv3/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block5_conv1/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block5_conv1/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block5_conv1/Conv2D/ReadVariableOp3VGG16_like/vgg16/block5_conv1/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block5_conv2/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block5_conv2/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block5_conv2/Conv2D/ReadVariableOp3VGG16_like/vgg16/block5_conv2/Conv2D/ReadVariableOp2l
4VGG16_like/vgg16/block5_conv3/BiasAdd/ReadVariableOp4VGG16_like/vgg16/block5_conv3/BiasAdd/ReadVariableOp2j
3VGG16_like/vgg16/block5_conv3/Conv2D/ReadVariableOp3VGG16_like/vgg16/block5_conv3/Conv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg16_input
�]
�
?__inference_vgg16_layer_call_and_return_conditional_losses_1348
input_1+
block1_conv1_1145:@
block1_conv1_1147:@+
block1_conv2_1161:@@
block1_conv2_1163:@,
block2_conv1_1178:@� 
block2_conv1_1180:	�-
block2_conv2_1194:�� 
block2_conv2_1196:	�-
block3_conv1_1211:�� 
block3_conv1_1213:	�-
block3_conv2_1227:�� 
block3_conv2_1229:	�-
block3_conv3_1243:�� 
block3_conv3_1245:	�-
block4_conv1_1260:�� 
block4_conv1_1262:	�-
block4_conv2_1276:�� 
block4_conv2_1278:	�-
block4_conv3_1292:�� 
block4_conv3_1294:	�-
block5_conv1_1309:�� 
block5_conv1_1311:	�-
block5_conv2_1325:�� 
block5_conv2_1327:	�-
block5_conv3_1341:�� 
block5_conv3_1343:	�
identity��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�$block2_conv1/StatefulPartitionedCall�$block2_conv2/StatefulPartitionedCall�$block3_conv1/StatefulPartitionedCall�$block3_conv2/StatefulPartitionedCall�$block3_conv3/StatefulPartitionedCall�$block4_conv1/StatefulPartitionedCall�$block4_conv2/StatefulPartitionedCall�$block4_conv3/StatefulPartitionedCall�$block5_conv1/StatefulPartitionedCall�$block5_conv2/StatefulPartitionedCall�$block5_conv3/StatefulPartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_1145block1_conv1_1147*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_1144�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_1161block1_conv2_1163*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_1160�
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_1086�
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_1178block2_conv1_1180*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_1177�
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_1194block2_conv2_1196*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_1193�
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_1096�
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_1211block3_conv1_1213*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_1210�
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_1227block3_conv2_1229*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_1226�
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_1243block3_conv3_1245*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_1242�
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_1106�
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_1260block4_conv1_1262*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_1259�
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_1276block4_conv2_1278*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_1275�
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_1292block4_conv3_1294*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_1291�
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_1116�
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_1309block5_conv1_1311*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_1308�
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_1325block5_conv2_1327*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_1324�
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_1341block5_conv3_1343*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_1340�
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_1126|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:$ 

_user_specified_name1343:$ 

_user_specified_name1341:$ 

_user_specified_name1327:$ 

_user_specified_name1325:$ 

_user_specified_name1311:$ 

_user_specified_name1309:$ 

_user_specified_name1294:$ 

_user_specified_name1292:$ 

_user_specified_name1278:$ 

_user_specified_name1276:$ 

_user_specified_name1262:$ 

_user_specified_name1260:$ 

_user_specified_name1245:$ 

_user_specified_name1243:$ 

_user_specified_name1229:$ 

_user_specified_name1227:$
 

_user_specified_name1213:$	 

_user_specified_name1211:$ 

_user_specified_name1196:$ 

_user_specified_name1194:$ 

_user_specified_name1180:$ 

_user_specified_name1178:$ 

_user_specified_name1163:$ 

_user_specified_name1161:$ 

_user_specified_name1147:$ 

_user_specified_name1145:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
+__inference_block3_conv1_layer_call_fn_2254

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_1210x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������88�<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2250:$ 

_user_specified_name2248:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
B
&__inference_flatten_layer_call_fn_2072

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1714b
IdentityIdentityPartitionedCall:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_block2_conv1_layer_call_and_return_conditional_losses_1177

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������pp�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������pp�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������pp�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������pp@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:W S
/
_output_shapes
:���������pp@
 
_user_specified_nameinputs
�
`
B__inference_dropout1_layer_call_and_return_conditional_losses_1823

inputs

identity_1P
IdentityIdentityinputs*
T0*)
_output_shapes
:�����������]

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:�����������"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2285

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
$__inference_vgg16_layer_call_fn_1479
input_1!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_1348x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name1475:$ 

_user_specified_name1473:$ 

_user_specified_name1471:$ 

_user_specified_name1469:$ 

_user_specified_name1467:$ 

_user_specified_name1465:$ 

_user_specified_name1463:$ 

_user_specified_name1461:$ 

_user_specified_name1459:$ 

_user_specified_name1457:$ 

_user_specified_name1455:$ 

_user_specified_name1453:$ 

_user_specified_name1451:$ 

_user_specified_name1449:$ 

_user_specified_name1447:$ 

_user_specified_name1445:$
 

_user_specified_name1443:$	 

_user_specified_name1441:$ 

_user_specified_name1439:$ 

_user_specified_name1437:$ 

_user_specified_name1435:$ 

_user_specified_name1433:$ 

_user_specified_name1431:$ 

_user_specified_name1429:$ 

_user_specified_name1427:$ 

_user_specified_name1425:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
�
�
)__inference_VGG16_like_layer_call_fn_1901
vgg16_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�

unknown_25:���

unknown_26:	�

unknown_27:	�

unknown_28:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvgg16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_VGG16_like_layer_call_and_return_conditional_losses_1762o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name1897:$ 

_user_specified_name1895:$ 

_user_specified_name1893:$ 

_user_specified_name1891:$ 

_user_specified_name1889:$ 

_user_specified_name1887:$ 

_user_specified_name1885:$ 

_user_specified_name1883:$ 

_user_specified_name1881:$ 

_user_specified_name1879:$ 

_user_specified_name1877:$ 

_user_specified_name1875:$ 

_user_specified_name1873:$ 

_user_specified_name1871:$ 

_user_specified_name1869:$ 

_user_specified_name1867:$ 

_user_specified_name1865:$ 

_user_specified_name1863:$ 

_user_specified_name1861:$ 

_user_specified_name1859:$
 

_user_specified_name1857:$	 

_user_specified_name1855:$ 

_user_specified_name1853:$ 

_user_specified_name1851:$ 

_user_specified_name1849:$ 

_user_specified_name1847:$ 

_user_specified_name1845:$ 

_user_specified_name1843:$ 

_user_specified_name1841:$ 

_user_specified_name1839:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg16_input
�
`
'__inference_dropout1_layer_call_fn_2083

inputs
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout1_layer_call_and_return_conditional_losses_1727q
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*)
_output_shapes
:�����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������22
StatefulPartitionedCallStatefulPartitionedCall:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�

a
B__inference_dropout1_layer_call_and_return_conditional_losses_1727

inputs
identity�R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *n۶?f
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:�����������Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::���
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:�����������*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *���>�
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:�����������T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    �
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*)
_output_shapes
:�����������c
IdentityIdentitydropout/SelectV2:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:�����������:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2375

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_block5_conv1_layer_call_fn_2394

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_1308x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2390:$ 

_user_specified_name2388:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
a
E__inference_block3_pool_layer_call_and_return_conditional_losses_2315

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
]
A__inference_flatten_layer_call_and_return_conditional_losses_2078

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� b  ^
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:�����������Z
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:�����������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:����������:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
F
*__inference_block4_pool_layer_call_fn_2380

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_1116�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
�
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2165

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������
 
_user_specified_nameinputs
�]
�
?__inference_vgg16_layer_call_and_return_conditional_losses_1422
input_1+
block1_conv1_1351:@
block1_conv1_1353:@+
block1_conv2_1356:@@
block1_conv2_1358:@,
block2_conv1_1362:@� 
block2_conv1_1364:	�-
block2_conv2_1367:�� 
block2_conv2_1369:	�-
block3_conv1_1373:�� 
block3_conv1_1375:	�-
block3_conv2_1378:�� 
block3_conv2_1380:	�-
block3_conv3_1383:�� 
block3_conv3_1385:	�-
block4_conv1_1389:�� 
block4_conv1_1391:	�-
block4_conv2_1394:�� 
block4_conv2_1396:	�-
block4_conv3_1399:�� 
block4_conv3_1401:	�-
block5_conv1_1405:�� 
block5_conv1_1407:	�-
block5_conv2_1410:�� 
block5_conv2_1412:	�-
block5_conv3_1415:�� 
block5_conv3_1417:	�
identity��$block1_conv1/StatefulPartitionedCall�$block1_conv2/StatefulPartitionedCall�$block2_conv1/StatefulPartitionedCall�$block2_conv2/StatefulPartitionedCall�$block3_conv1/StatefulPartitionedCall�$block3_conv2/StatefulPartitionedCall�$block3_conv3/StatefulPartitionedCall�$block4_conv1/StatefulPartitionedCall�$block4_conv2/StatefulPartitionedCall�$block4_conv3/StatefulPartitionedCall�$block5_conv1/StatefulPartitionedCall�$block5_conv2/StatefulPartitionedCall�$block5_conv3/StatefulPartitionedCall�
$block1_conv1/StatefulPartitionedCallStatefulPartitionedCallinput_1block1_conv1_1351block1_conv1_1353*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block1_conv1_layer_call_and_return_conditional_losses_1144�
$block1_conv2/StatefulPartitionedCallStatefulPartitionedCall-block1_conv1/StatefulPartitionedCall:output:0block1_conv2_1356block1_conv2_1358*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_1160�
block1_pool/PartitionedCallPartitionedCall-block1_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������pp@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_1086�
$block2_conv1/StatefulPartitionedCallStatefulPartitionedCall$block1_pool/PartitionedCall:output:0block2_conv1_1362block2_conv1_1364*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block2_conv1_layer_call_and_return_conditional_losses_1177�
$block2_conv2/StatefulPartitionedCallStatefulPartitionedCall-block2_conv1/StatefulPartitionedCall:output:0block2_conv2_1367block2_conv2_1369*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������pp�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block2_conv2_layer_call_and_return_conditional_losses_1193�
block2_pool/PartitionedCallPartitionedCall-block2_conv2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block2_pool_layer_call_and_return_conditional_losses_1096�
$block3_conv1/StatefulPartitionedCallStatefulPartitionedCall$block2_pool/PartitionedCall:output:0block3_conv1_1373block3_conv1_1375*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block3_conv1_layer_call_and_return_conditional_losses_1210�
$block3_conv2/StatefulPartitionedCallStatefulPartitionedCall-block3_conv1/StatefulPartitionedCall:output:0block3_conv2_1378block3_conv2_1380*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block3_conv2_layer_call_and_return_conditional_losses_1226�
$block3_conv3/StatefulPartitionedCallStatefulPartitionedCall-block3_conv2/StatefulPartitionedCall:output:0block3_conv3_1383block3_conv3_1385*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������88�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block3_conv3_layer_call_and_return_conditional_losses_1242�
block3_pool/PartitionedCallPartitionedCall-block3_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block3_pool_layer_call_and_return_conditional_losses_1106�
$block4_conv1/StatefulPartitionedCallStatefulPartitionedCall$block3_pool/PartitionedCall:output:0block4_conv1_1389block4_conv1_1391*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block4_conv1_layer_call_and_return_conditional_losses_1259�
$block4_conv2/StatefulPartitionedCallStatefulPartitionedCall-block4_conv1/StatefulPartitionedCall:output:0block4_conv2_1394block4_conv2_1396*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block4_conv2_layer_call_and_return_conditional_losses_1275�
$block4_conv3/StatefulPartitionedCallStatefulPartitionedCall-block4_conv2/StatefulPartitionedCall:output:0block4_conv3_1399block4_conv3_1401*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block4_conv3_layer_call_and_return_conditional_losses_1291�
block4_pool/PartitionedCallPartitionedCall-block4_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block4_pool_layer_call_and_return_conditional_losses_1116�
$block5_conv1/StatefulPartitionedCallStatefulPartitionedCall$block4_pool/PartitionedCall:output:0block5_conv1_1405block5_conv1_1407*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block5_conv1_layer_call_and_return_conditional_losses_1308�
$block5_conv2/StatefulPartitionedCallStatefulPartitionedCall-block5_conv1/StatefulPartitionedCall:output:0block5_conv2_1410block5_conv2_1412*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block5_conv2_layer_call_and_return_conditional_losses_1324�
$block5_conv3/StatefulPartitionedCallStatefulPartitionedCall-block5_conv2/StatefulPartitionedCall:output:0block5_conv3_1415block5_conv3_1417*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block5_conv3_layer_call_and_return_conditional_losses_1340�
block5_pool/PartitionedCallPartitionedCall-block5_conv3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block5_pool_layer_call_and_return_conditional_losses_1126|
IdentityIdentity$block5_pool/PartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:�����������
NoOpNoOp%^block1_conv1/StatefulPartitionedCall%^block1_conv2/StatefulPartitionedCall%^block2_conv1/StatefulPartitionedCall%^block2_conv2/StatefulPartitionedCall%^block3_conv1/StatefulPartitionedCall%^block3_conv2/StatefulPartitionedCall%^block3_conv3/StatefulPartitionedCall%^block4_conv1/StatefulPartitionedCall%^block4_conv2/StatefulPartitionedCall%^block4_conv3/StatefulPartitionedCall%^block5_conv1/StatefulPartitionedCall%^block5_conv2/StatefulPartitionedCall%^block5_conv3/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*d
_input_shapesS
Q:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$block1_conv1/StatefulPartitionedCall$block1_conv1/StatefulPartitionedCall2L
$block1_conv2/StatefulPartitionedCall$block1_conv2/StatefulPartitionedCall2L
$block2_conv1/StatefulPartitionedCall$block2_conv1/StatefulPartitionedCall2L
$block2_conv2/StatefulPartitionedCall$block2_conv2/StatefulPartitionedCall2L
$block3_conv1/StatefulPartitionedCall$block3_conv1/StatefulPartitionedCall2L
$block3_conv2/StatefulPartitionedCall$block3_conv2/StatefulPartitionedCall2L
$block3_conv3/StatefulPartitionedCall$block3_conv3/StatefulPartitionedCall2L
$block4_conv1/StatefulPartitionedCall$block4_conv1/StatefulPartitionedCall2L
$block4_conv2/StatefulPartitionedCall$block4_conv2/StatefulPartitionedCall2L
$block4_conv3/StatefulPartitionedCall$block4_conv3/StatefulPartitionedCall2L
$block5_conv1/StatefulPartitionedCall$block5_conv1/StatefulPartitionedCall2L
$block5_conv2/StatefulPartitionedCall$block5_conv2/StatefulPartitionedCall2L
$block5_conv3/StatefulPartitionedCall$block5_conv3/StatefulPartitionedCall:$ 

_user_specified_name1417:$ 

_user_specified_name1415:$ 

_user_specified_name1412:$ 

_user_specified_name1410:$ 

_user_specified_name1407:$ 

_user_specified_name1405:$ 

_user_specified_name1401:$ 

_user_specified_name1399:$ 

_user_specified_name1396:$ 

_user_specified_name1394:$ 

_user_specified_name1391:$ 

_user_specified_name1389:$ 

_user_specified_name1385:$ 

_user_specified_name1383:$ 

_user_specified_name1380:$ 

_user_specified_name1378:$
 

_user_specified_name1375:$	 

_user_specified_name1373:$ 

_user_specified_name1369:$ 

_user_specified_name1367:$ 

_user_specified_name1364:$ 

_user_specified_name1362:$ 

_user_specified_name1358:$ 

_user_specified_name1356:$ 

_user_specified_name1353:$ 

_user_specified_name1351:Z V
1
_output_shapes
:�����������
!
_user_specified_name	input_1
��
�Y
__inference__traced_save_3053
file_prefix9
$read_disablecopyonread_dense1_kernel:���3
$read_1_disablecopyonread_dense1_bias:	�8
%read_2_disablecopyonread_final_kernel:	�1
#read_3_disablecopyonread_final_bias:F
,read_4_disablecopyonread_block1_conv1_kernel:@8
*read_5_disablecopyonread_block1_conv1_bias:@F
,read_6_disablecopyonread_block1_conv2_kernel:@@8
*read_7_disablecopyonread_block1_conv2_bias:@G
,read_8_disablecopyonread_block2_conv1_kernel:@�9
*read_9_disablecopyonread_block2_conv1_bias:	�I
-read_10_disablecopyonread_block2_conv2_kernel:��:
+read_11_disablecopyonread_block2_conv2_bias:	�I
-read_12_disablecopyonread_block3_conv1_kernel:��:
+read_13_disablecopyonread_block3_conv1_bias:	�I
-read_14_disablecopyonread_block3_conv2_kernel:��:
+read_15_disablecopyonread_block3_conv2_bias:	�I
-read_16_disablecopyonread_block3_conv3_kernel:��:
+read_17_disablecopyonread_block3_conv3_bias:	�I
-read_18_disablecopyonread_block4_conv1_kernel:��:
+read_19_disablecopyonread_block4_conv1_bias:	�I
-read_20_disablecopyonread_block4_conv2_kernel:��:
+read_21_disablecopyonread_block4_conv2_bias:	�I
-read_22_disablecopyonread_block4_conv3_kernel:��:
+read_23_disablecopyonread_block4_conv3_bias:	�I
-read_24_disablecopyonread_block5_conv1_kernel:��:
+read_25_disablecopyonread_block5_conv1_bias:	�I
-read_26_disablecopyonread_block5_conv2_kernel:��:
+read_27_disablecopyonread_block5_conv2_bias:	�I
-read_28_disablecopyonread_block5_conv3_kernel:��:
+read_29_disablecopyonread_block5_conv3_bias:	�-
#read_30_disablecopyonread_iteration:	 1
'read_31_disablecopyonread_learning_rate: I
/read_32_disablecopyonread_m_block1_conv1_kernel:@I
/read_33_disablecopyonread_v_block1_conv1_kernel:@;
-read_34_disablecopyonread_m_block1_conv1_bias:@;
-read_35_disablecopyonread_v_block1_conv1_bias:@I
/read_36_disablecopyonread_m_block1_conv2_kernel:@@I
/read_37_disablecopyonread_v_block1_conv2_kernel:@@;
-read_38_disablecopyonread_m_block1_conv2_bias:@;
-read_39_disablecopyonread_v_block1_conv2_bias:@J
/read_40_disablecopyonread_m_block2_conv1_kernel:@�J
/read_41_disablecopyonread_v_block2_conv1_kernel:@�<
-read_42_disablecopyonread_m_block2_conv1_bias:	�<
-read_43_disablecopyonread_v_block2_conv1_bias:	�K
/read_44_disablecopyonread_m_block2_conv2_kernel:��K
/read_45_disablecopyonread_v_block2_conv2_kernel:��<
-read_46_disablecopyonread_m_block2_conv2_bias:	�<
-read_47_disablecopyonread_v_block2_conv2_bias:	�K
/read_48_disablecopyonread_m_block3_conv1_kernel:��K
/read_49_disablecopyonread_v_block3_conv1_kernel:��<
-read_50_disablecopyonread_m_block3_conv1_bias:	�<
-read_51_disablecopyonread_v_block3_conv1_bias:	�K
/read_52_disablecopyonread_m_block3_conv2_kernel:��K
/read_53_disablecopyonread_v_block3_conv2_kernel:��<
-read_54_disablecopyonread_m_block3_conv2_bias:	�<
-read_55_disablecopyonread_v_block3_conv2_bias:	�K
/read_56_disablecopyonread_m_block3_conv3_kernel:��K
/read_57_disablecopyonread_v_block3_conv3_kernel:��<
-read_58_disablecopyonread_m_block3_conv3_bias:	�<
-read_59_disablecopyonread_v_block3_conv3_bias:	�K
/read_60_disablecopyonread_m_block4_conv1_kernel:��K
/read_61_disablecopyonread_v_block4_conv1_kernel:��<
-read_62_disablecopyonread_m_block4_conv1_bias:	�<
-read_63_disablecopyonread_v_block4_conv1_bias:	�K
/read_64_disablecopyonread_m_block4_conv2_kernel:��K
/read_65_disablecopyonread_v_block4_conv2_kernel:��<
-read_66_disablecopyonread_m_block4_conv2_bias:	�<
-read_67_disablecopyonread_v_block4_conv2_bias:	�K
/read_68_disablecopyonread_m_block4_conv3_kernel:��K
/read_69_disablecopyonread_v_block4_conv3_kernel:��<
-read_70_disablecopyonread_m_block4_conv3_bias:	�<
-read_71_disablecopyonread_v_block4_conv3_bias:	�K
/read_72_disablecopyonread_m_block5_conv1_kernel:��K
/read_73_disablecopyonread_v_block5_conv1_kernel:��<
-read_74_disablecopyonread_m_block5_conv1_bias:	�<
-read_75_disablecopyonread_v_block5_conv1_bias:	�K
/read_76_disablecopyonread_m_block5_conv2_kernel:��K
/read_77_disablecopyonread_v_block5_conv2_kernel:��<
-read_78_disablecopyonread_m_block5_conv2_bias:	�<
-read_79_disablecopyonread_v_block5_conv2_bias:	�K
/read_80_disablecopyonread_m_block5_conv3_kernel:��K
/read_81_disablecopyonread_v_block5_conv3_kernel:��<
-read_82_disablecopyonread_m_block5_conv3_bias:	�<
-read_83_disablecopyonread_v_block5_conv3_bias:	�>
)read_84_disablecopyonread_m_dense1_kernel:���>
)read_85_disablecopyonread_v_dense1_kernel:���6
'read_86_disablecopyonread_m_dense1_bias:	�6
'read_87_disablecopyonread_v_dense1_bias:	�;
(read_88_disablecopyonread_m_final_kernel:	�;
(read_89_disablecopyonread_v_final_kernel:	�4
&read_90_disablecopyonread_m_final_bias:4
&read_91_disablecopyonread_v_final_bias:+
!read_92_disablecopyonread_total_1: +
!read_93_disablecopyonread_count_1: )
read_94_disablecopyonread_total: )
read_95_disablecopyonread_count: 
savev2_const
identity_193��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_27/DisableCopyOnRead�Read_27/ReadVariableOp�Read_28/DisableCopyOnRead�Read_28/ReadVariableOp�Read_29/DisableCopyOnRead�Read_29/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_30/DisableCopyOnRead�Read_30/ReadVariableOp�Read_31/DisableCopyOnRead�Read_31/ReadVariableOp�Read_32/DisableCopyOnRead�Read_32/ReadVariableOp�Read_33/DisableCopyOnRead�Read_33/ReadVariableOp�Read_34/DisableCopyOnRead�Read_34/ReadVariableOp�Read_35/DisableCopyOnRead�Read_35/ReadVariableOp�Read_36/DisableCopyOnRead�Read_36/ReadVariableOp�Read_37/DisableCopyOnRead�Read_37/ReadVariableOp�Read_38/DisableCopyOnRead�Read_38/ReadVariableOp�Read_39/DisableCopyOnRead�Read_39/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_40/DisableCopyOnRead�Read_40/ReadVariableOp�Read_41/DisableCopyOnRead�Read_41/ReadVariableOp�Read_42/DisableCopyOnRead�Read_42/ReadVariableOp�Read_43/DisableCopyOnRead�Read_43/ReadVariableOp�Read_44/DisableCopyOnRead�Read_44/ReadVariableOp�Read_45/DisableCopyOnRead�Read_45/ReadVariableOp�Read_46/DisableCopyOnRead�Read_46/ReadVariableOp�Read_47/DisableCopyOnRead�Read_47/ReadVariableOp�Read_48/DisableCopyOnRead�Read_48/ReadVariableOp�Read_49/DisableCopyOnRead�Read_49/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_50/DisableCopyOnRead�Read_50/ReadVariableOp�Read_51/DisableCopyOnRead�Read_51/ReadVariableOp�Read_52/DisableCopyOnRead�Read_52/ReadVariableOp�Read_53/DisableCopyOnRead�Read_53/ReadVariableOp�Read_54/DisableCopyOnRead�Read_54/ReadVariableOp�Read_55/DisableCopyOnRead�Read_55/ReadVariableOp�Read_56/DisableCopyOnRead�Read_56/ReadVariableOp�Read_57/DisableCopyOnRead�Read_57/ReadVariableOp�Read_58/DisableCopyOnRead�Read_58/ReadVariableOp�Read_59/DisableCopyOnRead�Read_59/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_60/DisableCopyOnRead�Read_60/ReadVariableOp�Read_61/DisableCopyOnRead�Read_61/ReadVariableOp�Read_62/DisableCopyOnRead�Read_62/ReadVariableOp�Read_63/DisableCopyOnRead�Read_63/ReadVariableOp�Read_64/DisableCopyOnRead�Read_64/ReadVariableOp�Read_65/DisableCopyOnRead�Read_65/ReadVariableOp�Read_66/DisableCopyOnRead�Read_66/ReadVariableOp�Read_67/DisableCopyOnRead�Read_67/ReadVariableOp�Read_68/DisableCopyOnRead�Read_68/ReadVariableOp�Read_69/DisableCopyOnRead�Read_69/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_70/DisableCopyOnRead�Read_70/ReadVariableOp�Read_71/DisableCopyOnRead�Read_71/ReadVariableOp�Read_72/DisableCopyOnRead�Read_72/ReadVariableOp�Read_73/DisableCopyOnRead�Read_73/ReadVariableOp�Read_74/DisableCopyOnRead�Read_74/ReadVariableOp�Read_75/DisableCopyOnRead�Read_75/ReadVariableOp�Read_76/DisableCopyOnRead�Read_76/ReadVariableOp�Read_77/DisableCopyOnRead�Read_77/ReadVariableOp�Read_78/DisableCopyOnRead�Read_78/ReadVariableOp�Read_79/DisableCopyOnRead�Read_79/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_80/DisableCopyOnRead�Read_80/ReadVariableOp�Read_81/DisableCopyOnRead�Read_81/ReadVariableOp�Read_82/DisableCopyOnRead�Read_82/ReadVariableOp�Read_83/DisableCopyOnRead�Read_83/ReadVariableOp�Read_84/DisableCopyOnRead�Read_84/ReadVariableOp�Read_85/DisableCopyOnRead�Read_85/ReadVariableOp�Read_86/DisableCopyOnRead�Read_86/ReadVariableOp�Read_87/DisableCopyOnRead�Read_87/ReadVariableOp�Read_88/DisableCopyOnRead�Read_88/ReadVariableOp�Read_89/DisableCopyOnRead�Read_89/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOp�Read_90/DisableCopyOnRead�Read_90/ReadVariableOp�Read_91/DisableCopyOnRead�Read_91/ReadVariableOp�Read_92/DisableCopyOnRead�Read_92/ReadVariableOp�Read_93/DisableCopyOnRead�Read_93/ReadVariableOp�Read_94/DisableCopyOnRead�Read_94/ReadVariableOp�Read_95/DisableCopyOnRead�Read_95/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: v
Read/DisableCopyOnReadDisableCopyOnRead$read_disablecopyonread_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp$read_disablecopyonread_dense1_kernel^Read/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:���*
dtype0l
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:���d

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*!
_output_shapes
:���x
Read_1/DisableCopyOnReadDisableCopyOnRead$read_1_disablecopyonread_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp$read_1_disablecopyonread_dense1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0j

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�`

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes	
:�y
Read_2/DisableCopyOnReadDisableCopyOnRead%read_2_disablecopyonread_final_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp%read_2_disablecopyonread_final_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0n

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�d

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes
:	�w
Read_3/DisableCopyOnReadDisableCopyOnRead#read_3_disablecopyonread_final_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp#read_3_disablecopyonread_final_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:�
Read_4/DisableCopyOnReadDisableCopyOnRead,read_4_disablecopyonread_block1_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp,read_4_disablecopyonread_block1_conv1_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0u

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@k

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*&
_output_shapes
:@~
Read_5/DisableCopyOnReadDisableCopyOnRead*read_5_disablecopyonread_block1_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp*read_5_disablecopyonread_block1_conv1_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_6/DisableCopyOnReadDisableCopyOnRead,read_6_disablecopyonread_block1_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp,read_6_disablecopyonread_block1_conv2_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0v
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@~
Read_7/DisableCopyOnReadDisableCopyOnRead*read_7_disablecopyonread_block1_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp*read_7_disablecopyonread_block1_conv2_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_8/DisableCopyOnReadDisableCopyOnRead,read_8_disablecopyonread_block2_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp,read_8_disablecopyonread_block2_conv1_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0w
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*'
_output_shapes
:@�~
Read_9/DisableCopyOnReadDisableCopyOnRead*read_9_disablecopyonread_block2_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp*read_9_disablecopyonread_block2_conv1_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0k
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_10/DisableCopyOnReadDisableCopyOnRead-read_10_disablecopyonread_block2_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp-read_10_disablecopyonread_block2_conv2_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_11/DisableCopyOnReadDisableCopyOnRead+read_11_disablecopyonread_block2_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp+read_11_disablecopyonread_block2_conv2_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_12/DisableCopyOnReadDisableCopyOnRead-read_12_disablecopyonread_block3_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp-read_12_disablecopyonread_block3_conv1_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_13/DisableCopyOnReadDisableCopyOnRead+read_13_disablecopyonread_block3_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp+read_13_disablecopyonread_block3_conv1_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_14/DisableCopyOnReadDisableCopyOnRead-read_14_disablecopyonread_block3_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp-read_14_disablecopyonread_block3_conv2_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_15/DisableCopyOnReadDisableCopyOnRead+read_15_disablecopyonread_block3_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp+read_15_disablecopyonread_block3_conv2_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_16/DisableCopyOnReadDisableCopyOnRead-read_16_disablecopyonread_block3_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp-read_16_disablecopyonread_block3_conv3_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_17/DisableCopyOnReadDisableCopyOnRead+read_17_disablecopyonread_block3_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp+read_17_disablecopyonread_block3_conv3_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_18/DisableCopyOnReadDisableCopyOnRead-read_18_disablecopyonread_block4_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp-read_18_disablecopyonread_block4_conv1_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_19/DisableCopyOnReadDisableCopyOnRead+read_19_disablecopyonread_block4_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp+read_19_disablecopyonread_block4_conv1_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_20/DisableCopyOnReadDisableCopyOnRead-read_20_disablecopyonread_block4_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp-read_20_disablecopyonread_block4_conv2_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_21/DisableCopyOnReadDisableCopyOnRead+read_21_disablecopyonread_block4_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOp+read_21_disablecopyonread_block4_conv2_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_22/DisableCopyOnReadDisableCopyOnRead-read_22_disablecopyonread_block4_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOp-read_22_disablecopyonread_block4_conv3_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_23/DisableCopyOnReadDisableCopyOnRead+read_23_disablecopyonread_block4_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp+read_23_disablecopyonread_block4_conv3_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_24/DisableCopyOnReadDisableCopyOnRead-read_24_disablecopyonread_block5_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp-read_24_disablecopyonread_block5_conv1_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_25/DisableCopyOnReadDisableCopyOnRead+read_25_disablecopyonread_block5_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp+read_25_disablecopyonread_block5_conv1_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_26/DisableCopyOnReadDisableCopyOnRead-read_26_disablecopyonread_block5_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp-read_26_disablecopyonread_block5_conv2_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_27/DisableCopyOnReadDisableCopyOnRead+read_27_disablecopyonread_block5_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_27/ReadVariableOpReadVariableOp+read_27_disablecopyonread_block5_conv2_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_28/DisableCopyOnReadDisableCopyOnRead-read_28_disablecopyonread_block5_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_28/ReadVariableOpReadVariableOp-read_28_disablecopyonread_block5_conv3_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_29/DisableCopyOnReadDisableCopyOnRead+read_29_disablecopyonread_block5_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_29/ReadVariableOpReadVariableOp+read_29_disablecopyonread_block5_conv3_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes	
:�x
Read_30/DisableCopyOnReadDisableCopyOnRead#read_30_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 �
Read_30/ReadVariableOpReadVariableOp#read_30_disablecopyonread_iteration^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_31/DisableCopyOnReadDisableCopyOnRead'read_31_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 �
Read_31/ReadVariableOpReadVariableOp'read_31_disablecopyonread_learning_rate^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
: �
Read_32/DisableCopyOnReadDisableCopyOnRead/read_32_disablecopyonread_m_block1_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_32/ReadVariableOpReadVariableOp/read_32_disablecopyonread_m_block1_conv1_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_33/DisableCopyOnReadDisableCopyOnRead/read_33_disablecopyonread_v_block1_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_33/ReadVariableOpReadVariableOp/read_33_disablecopyonread_v_block1_conv1_kernel^Read_33/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@*
dtype0w
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@m
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*&
_output_shapes
:@�
Read_34/DisableCopyOnReadDisableCopyOnRead-read_34_disablecopyonread_m_block1_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_34/ReadVariableOpReadVariableOp-read_34_disablecopyonread_m_block1_conv1_bias^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_35/DisableCopyOnReadDisableCopyOnRead-read_35_disablecopyonread_v_block1_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_35/ReadVariableOpReadVariableOp-read_35_disablecopyonread_v_block1_conv1_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_36/DisableCopyOnReadDisableCopyOnRead/read_36_disablecopyonread_m_block1_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_36/ReadVariableOpReadVariableOp/read_36_disablecopyonread_m_block1_conv2_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_37/DisableCopyOnReadDisableCopyOnRead/read_37_disablecopyonread_v_block1_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_37/ReadVariableOpReadVariableOp/read_37_disablecopyonread_v_block1_conv2_kernel^Read_37/DisableCopyOnRead"/device:CPU:0*&
_output_shapes
:@@*
dtype0w
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*&
_output_shapes
:@@m
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*&
_output_shapes
:@@�
Read_38/DisableCopyOnReadDisableCopyOnRead-read_38_disablecopyonread_m_block1_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_38/ReadVariableOpReadVariableOp-read_38_disablecopyonread_m_block1_conv2_bias^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_39/DisableCopyOnReadDisableCopyOnRead-read_39_disablecopyonread_v_block1_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_39/ReadVariableOpReadVariableOp-read_39_disablecopyonread_v_block1_conv2_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_40/DisableCopyOnReadDisableCopyOnRead/read_40_disablecopyonread_m_block2_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_40/ReadVariableOpReadVariableOp/read_40_disablecopyonread_m_block2_conv1_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_41/DisableCopyOnReadDisableCopyOnRead/read_41_disablecopyonread_v_block2_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_41/ReadVariableOpReadVariableOp/read_41_disablecopyonread_v_block2_conv1_kernel^Read_41/DisableCopyOnRead"/device:CPU:0*'
_output_shapes
:@�*
dtype0x
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*'
_output_shapes
:@�n
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*'
_output_shapes
:@��
Read_42/DisableCopyOnReadDisableCopyOnRead-read_42_disablecopyonread_m_block2_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_42/ReadVariableOpReadVariableOp-read_42_disablecopyonread_m_block2_conv1_bias^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_43/DisableCopyOnReadDisableCopyOnRead-read_43_disablecopyonread_v_block2_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_43/ReadVariableOpReadVariableOp-read_43_disablecopyonread_v_block2_conv1_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_44/DisableCopyOnReadDisableCopyOnRead/read_44_disablecopyonread_m_block2_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_44/ReadVariableOpReadVariableOp/read_44_disablecopyonread_m_block2_conv2_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_45/DisableCopyOnReadDisableCopyOnRead/read_45_disablecopyonread_v_block2_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_45/ReadVariableOpReadVariableOp/read_45_disablecopyonread_v_block2_conv2_kernel^Read_45/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_46/DisableCopyOnReadDisableCopyOnRead-read_46_disablecopyonread_m_block2_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_46/ReadVariableOpReadVariableOp-read_46_disablecopyonread_m_block2_conv2_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_47/DisableCopyOnReadDisableCopyOnRead-read_47_disablecopyonread_v_block2_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_47/ReadVariableOpReadVariableOp-read_47_disablecopyonread_v_block2_conv2_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0l
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�b
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_48/DisableCopyOnReadDisableCopyOnRead/read_48_disablecopyonread_m_block3_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_48/ReadVariableOpReadVariableOp/read_48_disablecopyonread_m_block3_conv1_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_49/DisableCopyOnReadDisableCopyOnRead/read_49_disablecopyonread_v_block3_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_49/ReadVariableOpReadVariableOp/read_49_disablecopyonread_v_block3_conv1_kernel^Read_49/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0y
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��o
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_50/DisableCopyOnReadDisableCopyOnRead-read_50_disablecopyonread_m_block3_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_50/ReadVariableOpReadVariableOp-read_50_disablecopyonread_m_block3_conv1_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_51/DisableCopyOnReadDisableCopyOnRead-read_51_disablecopyonread_v_block3_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_51/ReadVariableOpReadVariableOp-read_51_disablecopyonread_v_block3_conv1_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_52/DisableCopyOnReadDisableCopyOnRead/read_52_disablecopyonread_m_block3_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_52/ReadVariableOpReadVariableOp/read_52_disablecopyonread_m_block3_conv2_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_53/DisableCopyOnReadDisableCopyOnRead/read_53_disablecopyonread_v_block3_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_53/ReadVariableOpReadVariableOp/read_53_disablecopyonread_v_block3_conv2_kernel^Read_53/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_54/DisableCopyOnReadDisableCopyOnRead-read_54_disablecopyonread_m_block3_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_54/ReadVariableOpReadVariableOp-read_54_disablecopyonread_m_block3_conv2_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_55/DisableCopyOnReadDisableCopyOnRead-read_55_disablecopyonread_v_block3_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_55/ReadVariableOpReadVariableOp-read_55_disablecopyonread_v_block3_conv2_bias^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_56/DisableCopyOnReadDisableCopyOnRead/read_56_disablecopyonread_m_block3_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_56/ReadVariableOpReadVariableOp/read_56_disablecopyonread_m_block3_conv3_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_57/DisableCopyOnReadDisableCopyOnRead/read_57_disablecopyonread_v_block3_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_57/ReadVariableOpReadVariableOp/read_57_disablecopyonread_v_block3_conv3_kernel^Read_57/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_58/DisableCopyOnReadDisableCopyOnRead-read_58_disablecopyonread_m_block3_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_58/ReadVariableOpReadVariableOp-read_58_disablecopyonread_m_block3_conv3_bias^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_59/DisableCopyOnReadDisableCopyOnRead-read_59_disablecopyonread_v_block3_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_59/ReadVariableOpReadVariableOp-read_59_disablecopyonread_v_block3_conv3_bias^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_60/DisableCopyOnReadDisableCopyOnRead/read_60_disablecopyonread_m_block4_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_60/ReadVariableOpReadVariableOp/read_60_disablecopyonread_m_block4_conv1_kernel^Read_60/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_61/DisableCopyOnReadDisableCopyOnRead/read_61_disablecopyonread_v_block4_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_61/ReadVariableOpReadVariableOp/read_61_disablecopyonread_v_block4_conv1_kernel^Read_61/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_62/DisableCopyOnReadDisableCopyOnRead-read_62_disablecopyonread_m_block4_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_62/ReadVariableOpReadVariableOp-read_62_disablecopyonread_m_block4_conv1_bias^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_63/DisableCopyOnReadDisableCopyOnRead-read_63_disablecopyonread_v_block4_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_63/ReadVariableOpReadVariableOp-read_63_disablecopyonread_v_block4_conv1_bias^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_64/DisableCopyOnReadDisableCopyOnRead/read_64_disablecopyonread_m_block4_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_64/ReadVariableOpReadVariableOp/read_64_disablecopyonread_m_block4_conv2_kernel^Read_64/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_65/DisableCopyOnReadDisableCopyOnRead/read_65_disablecopyonread_v_block4_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_65/ReadVariableOpReadVariableOp/read_65_disablecopyonread_v_block4_conv2_kernel^Read_65/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_66/DisableCopyOnReadDisableCopyOnRead-read_66_disablecopyonread_m_block4_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_66/ReadVariableOpReadVariableOp-read_66_disablecopyonread_m_block4_conv2_bias^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_67/DisableCopyOnReadDisableCopyOnRead-read_67_disablecopyonread_v_block4_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_67/ReadVariableOpReadVariableOp-read_67_disablecopyonread_v_block4_conv2_bias^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_68/DisableCopyOnReadDisableCopyOnRead/read_68_disablecopyonread_m_block4_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_68/ReadVariableOpReadVariableOp/read_68_disablecopyonread_m_block4_conv3_kernel^Read_68/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_69/DisableCopyOnReadDisableCopyOnRead/read_69_disablecopyonread_v_block4_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_69/ReadVariableOpReadVariableOp/read_69_disablecopyonread_v_block4_conv3_kernel^Read_69/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_70/DisableCopyOnReadDisableCopyOnRead-read_70_disablecopyonread_m_block4_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_70/ReadVariableOpReadVariableOp-read_70_disablecopyonread_m_block4_conv3_bias^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_71/DisableCopyOnReadDisableCopyOnRead-read_71_disablecopyonread_v_block4_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_71/ReadVariableOpReadVariableOp-read_71_disablecopyonread_v_block4_conv3_bias^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_72/DisableCopyOnReadDisableCopyOnRead/read_72_disablecopyonread_m_block5_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_72/ReadVariableOpReadVariableOp/read_72_disablecopyonread_m_block5_conv1_kernel^Read_72/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_73/DisableCopyOnReadDisableCopyOnRead/read_73_disablecopyonread_v_block5_conv1_kernel"/device:CPU:0*
_output_shapes
 �
Read_73/ReadVariableOpReadVariableOp/read_73_disablecopyonread_v_block5_conv1_kernel^Read_73/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_74/DisableCopyOnReadDisableCopyOnRead-read_74_disablecopyonread_m_block5_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_74/ReadVariableOpReadVariableOp-read_74_disablecopyonread_m_block5_conv1_bias^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_75/DisableCopyOnReadDisableCopyOnRead-read_75_disablecopyonread_v_block5_conv1_bias"/device:CPU:0*
_output_shapes
 �
Read_75/ReadVariableOpReadVariableOp-read_75_disablecopyonread_v_block5_conv1_bias^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_76/DisableCopyOnReadDisableCopyOnRead/read_76_disablecopyonread_m_block5_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_76/ReadVariableOpReadVariableOp/read_76_disablecopyonread_m_block5_conv2_kernel^Read_76/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_77/DisableCopyOnReadDisableCopyOnRead/read_77_disablecopyonread_v_block5_conv2_kernel"/device:CPU:0*
_output_shapes
 �
Read_77/ReadVariableOpReadVariableOp/read_77_disablecopyonread_v_block5_conv2_kernel^Read_77/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_78/DisableCopyOnReadDisableCopyOnRead-read_78_disablecopyonread_m_block5_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_78/ReadVariableOpReadVariableOp-read_78_disablecopyonread_m_block5_conv2_bias^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_79/DisableCopyOnReadDisableCopyOnRead-read_79_disablecopyonread_v_block5_conv2_bias"/device:CPU:0*
_output_shapes
 �
Read_79/ReadVariableOpReadVariableOp-read_79_disablecopyonread_v_block5_conv2_bias^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_80/DisableCopyOnReadDisableCopyOnRead/read_80_disablecopyonread_m_block5_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_80/ReadVariableOpReadVariableOp/read_80_disablecopyonread_m_block5_conv3_kernel^Read_80/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_81/DisableCopyOnReadDisableCopyOnRead/read_81_disablecopyonread_v_block5_conv3_kernel"/device:CPU:0*
_output_shapes
 �
Read_81/ReadVariableOpReadVariableOp/read_81_disablecopyonread_v_block5_conv3_kernel^Read_81/DisableCopyOnRead"/device:CPU:0*(
_output_shapes
:��*
dtype0z
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*(
_output_shapes
:��q
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*(
_output_shapes
:���
Read_82/DisableCopyOnReadDisableCopyOnRead-read_82_disablecopyonread_m_block5_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_82/ReadVariableOpReadVariableOp-read_82_disablecopyonread_m_block5_conv3_bias^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes	
:��
Read_83/DisableCopyOnReadDisableCopyOnRead-read_83_disablecopyonread_v_block5_conv3_bias"/device:CPU:0*
_output_shapes
 �
Read_83/ReadVariableOpReadVariableOp-read_83_disablecopyonread_v_block5_conv3_bias^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes	
:�~
Read_84/DisableCopyOnReadDisableCopyOnRead)read_84_disablecopyonread_m_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read_84/ReadVariableOpReadVariableOp)read_84_disablecopyonread_m_dense1_kernel^Read_84/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:���*
dtype0s
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:���j
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*!
_output_shapes
:���~
Read_85/DisableCopyOnReadDisableCopyOnRead)read_85_disablecopyonread_v_dense1_kernel"/device:CPU:0*
_output_shapes
 �
Read_85/ReadVariableOpReadVariableOp)read_85_disablecopyonread_v_dense1_kernel^Read_85/DisableCopyOnRead"/device:CPU:0*!
_output_shapes
:���*
dtype0s
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*!
_output_shapes
:���j
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*!
_output_shapes
:���|
Read_86/DisableCopyOnReadDisableCopyOnRead'read_86_disablecopyonread_m_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_86/ReadVariableOpReadVariableOp'read_86_disablecopyonread_m_dense1_bias^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes	
:�|
Read_87/DisableCopyOnReadDisableCopyOnRead'read_87_disablecopyonread_v_dense1_bias"/device:CPU:0*
_output_shapes
 �
Read_87/ReadVariableOpReadVariableOp'read_87_disablecopyonread_v_dense1_bias^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:�*
dtype0m
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:�d
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes	
:�}
Read_88/DisableCopyOnReadDisableCopyOnRead(read_88_disablecopyonread_m_final_kernel"/device:CPU:0*
_output_shapes
 �
Read_88/ReadVariableOpReadVariableOp(read_88_disablecopyonread_m_final_kernel^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes
:	�}
Read_89/DisableCopyOnReadDisableCopyOnRead(read_89_disablecopyonread_v_final_kernel"/device:CPU:0*
_output_shapes
 �
Read_89/ReadVariableOpReadVariableOp(read_89_disablecopyonread_v_final_kernel^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	�*
dtype0q
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	�h
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes
:	�{
Read_90/DisableCopyOnReadDisableCopyOnRead&read_90_disablecopyonread_m_final_bias"/device:CPU:0*
_output_shapes
 �
Read_90/ReadVariableOpReadVariableOp&read_90_disablecopyonread_m_final_bias^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes
:{
Read_91/DisableCopyOnReadDisableCopyOnRead&read_91_disablecopyonread_v_final_bias"/device:CPU:0*
_output_shapes
 �
Read_91/ReadVariableOpReadVariableOp&read_91_disablecopyonread_v_final_bias^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_92/DisableCopyOnReadDisableCopyOnRead!read_92_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 �
Read_92/ReadVariableOpReadVariableOp!read_92_disablecopyonread_total_1^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_93/DisableCopyOnReadDisableCopyOnRead!read_93_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 �
Read_93/ReadVariableOpReadVariableOp!read_93_disablecopyonread_count_1^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_94/DisableCopyOnReadDisableCopyOnReadread_94_disablecopyonread_total"/device:CPU:0*
_output_shapes
 �
Read_94/ReadVariableOpReadVariableOpread_94_disablecopyonread_total^Read_94/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes
: t
Read_95/DisableCopyOnReadDisableCopyOnReadread_95_disablecopyonread_count"/device:CPU:0*
_output_shapes
 �
Read_95/ReadVariableOpReadVariableOpread_95_disablecopyonread_count^Read_95/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*
_output_shapes
: �%
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*�%
value�%B�%aB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/55/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/56/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/57/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/58/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/59/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/60/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:a*
dtype0*�
value�B�aB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *o
dtypese
c2a	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_192Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_193IdentityIdentity_192:output:0^NoOp*
T0*
_output_shapes
: �'
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp*
_output_shapes
 "%
identity_193Identity_193:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp:=a9

_output_shapes
: 

_user_specified_nameConst:%`!

_user_specified_namecount:%_!

_user_specified_nametotal:'^#
!
_user_specified_name	count_1:']#
!
_user_specified_name	total_1:,\(
&
_user_specified_namev/final/bias:,[(
&
_user_specified_namem/final/bias:.Z*
(
_user_specified_namev/final/kernel:.Y*
(
_user_specified_namem/final/kernel:-X)
'
_user_specified_namev/dense1/bias:-W)
'
_user_specified_namem/dense1/bias:/V+
)
_user_specified_namev/dense1/kernel:/U+
)
_user_specified_namem/dense1/kernel:3T/
-
_user_specified_namev/block5_conv3/bias:3S/
-
_user_specified_namem/block5_conv3/bias:5R1
/
_user_specified_namev/block5_conv3/kernel:5Q1
/
_user_specified_namem/block5_conv3/kernel:3P/
-
_user_specified_namev/block5_conv2/bias:3O/
-
_user_specified_namem/block5_conv2/bias:5N1
/
_user_specified_namev/block5_conv2/kernel:5M1
/
_user_specified_namem/block5_conv2/kernel:3L/
-
_user_specified_namev/block5_conv1/bias:3K/
-
_user_specified_namem/block5_conv1/bias:5J1
/
_user_specified_namev/block5_conv1/kernel:5I1
/
_user_specified_namem/block5_conv1/kernel:3H/
-
_user_specified_namev/block4_conv3/bias:3G/
-
_user_specified_namem/block4_conv3/bias:5F1
/
_user_specified_namev/block4_conv3/kernel:5E1
/
_user_specified_namem/block4_conv3/kernel:3D/
-
_user_specified_namev/block4_conv2/bias:3C/
-
_user_specified_namem/block4_conv2/bias:5B1
/
_user_specified_namev/block4_conv2/kernel:5A1
/
_user_specified_namem/block4_conv2/kernel:3@/
-
_user_specified_namev/block4_conv1/bias:3?/
-
_user_specified_namem/block4_conv1/bias:5>1
/
_user_specified_namev/block4_conv1/kernel:5=1
/
_user_specified_namem/block4_conv1/kernel:3</
-
_user_specified_namev/block3_conv3/bias:3;/
-
_user_specified_namem/block3_conv3/bias:5:1
/
_user_specified_namev/block3_conv3/kernel:591
/
_user_specified_namem/block3_conv3/kernel:38/
-
_user_specified_namev/block3_conv2/bias:37/
-
_user_specified_namem/block3_conv2/bias:561
/
_user_specified_namev/block3_conv2/kernel:551
/
_user_specified_namem/block3_conv2/kernel:34/
-
_user_specified_namev/block3_conv1/bias:33/
-
_user_specified_namem/block3_conv1/bias:521
/
_user_specified_namev/block3_conv1/kernel:511
/
_user_specified_namem/block3_conv1/kernel:30/
-
_user_specified_namev/block2_conv2/bias:3//
-
_user_specified_namem/block2_conv2/bias:5.1
/
_user_specified_namev/block2_conv2/kernel:5-1
/
_user_specified_namem/block2_conv2/kernel:3,/
-
_user_specified_namev/block2_conv1/bias:3+/
-
_user_specified_namem/block2_conv1/bias:5*1
/
_user_specified_namev/block2_conv1/kernel:5)1
/
_user_specified_namem/block2_conv1/kernel:3(/
-
_user_specified_namev/block1_conv2/bias:3'/
-
_user_specified_namem/block1_conv2/bias:5&1
/
_user_specified_namev/block1_conv2/kernel:5%1
/
_user_specified_namem/block1_conv2/kernel:3$/
-
_user_specified_namev/block1_conv1/bias:3#/
-
_user_specified_namem/block1_conv1/bias:5"1
/
_user_specified_namev/block1_conv1/kernel:5!1
/
_user_specified_namem/block1_conv1/kernel:- )
'
_user_specified_namelearning_rate:)%
#
_user_specified_name	iteration:1-
+
_user_specified_nameblock5_conv3/bias:3/
-
_user_specified_nameblock5_conv3/kernel:1-
+
_user_specified_nameblock5_conv2/bias:3/
-
_user_specified_nameblock5_conv2/kernel:1-
+
_user_specified_nameblock5_conv1/bias:3/
-
_user_specified_nameblock5_conv1/kernel:1-
+
_user_specified_nameblock4_conv3/bias:3/
-
_user_specified_nameblock4_conv3/kernel:1-
+
_user_specified_nameblock4_conv2/bias:3/
-
_user_specified_nameblock4_conv2/kernel:1-
+
_user_specified_nameblock4_conv1/bias:3/
-
_user_specified_nameblock4_conv1/kernel:1-
+
_user_specified_nameblock3_conv3/bias:3/
-
_user_specified_nameblock3_conv3/kernel:1-
+
_user_specified_nameblock3_conv2/bias:3/
-
_user_specified_nameblock3_conv2/kernel:1-
+
_user_specified_nameblock3_conv1/bias:3/
-
_user_specified_nameblock3_conv1/kernel:1-
+
_user_specified_nameblock2_conv2/bias:3/
-
_user_specified_nameblock2_conv2/kernel:1
-
+
_user_specified_nameblock2_conv1/bias:3	/
-
_user_specified_nameblock2_conv1/kernel:1-
+
_user_specified_nameblock1_conv2/bias:3/
-
_user_specified_nameblock1_conv2/kernel:1-
+
_user_specified_nameblock1_conv1/bias:3/
-
_user_specified_nameblock1_conv1/kernel:*&
$
_user_specified_name
final/bias:,(
&
_user_specified_namefinal/kernel:+'
%
_user_specified_namedense1/bias:-)
'
_user_specified_namedense1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_2455

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�

�
@__inference_dense1_layer_call_and_return_conditional_losses_1739

inputs3
matmul_readvariableop_resource:���.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpw
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*!
_output_shapes
:���*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
)__inference_VGG16_like_layer_call_fn_1966
vgg16_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@$
	unknown_3:@�
	unknown_4:	�%
	unknown_5:��
	unknown_6:	�%
	unknown_7:��
	unknown_8:	�%
	unknown_9:��

unknown_10:	�&

unknown_11:��

unknown_12:	�&

unknown_13:��

unknown_14:	�&

unknown_15:��

unknown_16:	�&

unknown_17:��

unknown_18:	�&

unknown_19:��

unknown_20:	�&

unknown_21:��

unknown_22:	�&

unknown_23:��

unknown_24:	�

unknown_25:���

unknown_26:	�

unknown_27:	�

unknown_28:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallvgg16_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_VGG16_like_layer_call_and_return_conditional_losses_1836o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name1962:$ 

_user_specified_name1960:$ 

_user_specified_name1958:$ 

_user_specified_name1956:$ 

_user_specified_name1954:$ 

_user_specified_name1952:$ 

_user_specified_name1950:$ 

_user_specified_name1948:$ 

_user_specified_name1946:$ 

_user_specified_name1944:$ 

_user_specified_name1942:$ 

_user_specified_name1940:$ 

_user_specified_name1938:$ 

_user_specified_name1936:$ 

_user_specified_name1934:$ 

_user_specified_name1932:$ 

_user_specified_name1930:$ 

_user_specified_name1928:$ 

_user_specified_name1926:$ 

_user_specified_name1924:$
 

_user_specified_name1922:$	 

_user_specified_name1920:$ 

_user_specified_name1918:$ 

_user_specified_name1916:$ 

_user_specified_name1914:$ 

_user_specified_name1912:$ 

_user_specified_name1910:$ 

_user_specified_name1908:$ 

_user_specified_name1906:$ 

_user_specified_name1904:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg16_input
�
a
E__inference_block5_pool_layer_call_and_return_conditional_losses_1126

inputs
identity�
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�
F
*__inference_block1_pool_layer_call_fn_2190

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4������������������������������������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_block1_pool_layer_call_and_return_conditional_losses_1086�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4������������������������������������:r n
J
_output_shapes8
6:4������������������������������������
 
_user_specified_nameinputs
�*
�	
D__inference_VGG16_like_layer_call_and_return_conditional_losses_1762
vgg16_input$

vgg16_1656:@

vgg16_1658:@$

vgg16_1660:@@

vgg16_1662:@%

vgg16_1664:@�

vgg16_1666:	�&

vgg16_1668:��

vgg16_1670:	�&

vgg16_1672:��

vgg16_1674:	�&

vgg16_1676:��

vgg16_1678:	�&

vgg16_1680:��

vgg16_1682:	�&

vgg16_1684:��

vgg16_1686:	�&

vgg16_1688:��

vgg16_1690:	�&

vgg16_1692:��

vgg16_1694:	�&

vgg16_1696:��

vgg16_1698:	�&

vgg16_1700:��

vgg16_1702:	�&

vgg16_1704:��

vgg16_1706:	� 
dense1_1740:���
dense1_1742:	�

final_1756:	�

final_1758:
identity��dense1/StatefulPartitionedCall� dropout1/StatefulPartitionedCall�final/StatefulPartitionedCall�vgg16/StatefulPartitionedCall�
vgg16/StatefulPartitionedCallStatefulPartitionedCallvgg16_input
vgg16_1656
vgg16_1658
vgg16_1660
vgg16_1662
vgg16_1664
vgg16_1666
vgg16_1668
vgg16_1670
vgg16_1672
vgg16_1674
vgg16_1676
vgg16_1678
vgg16_1680
vgg16_1682
vgg16_1684
vgg16_1686
vgg16_1688
vgg16_1690
vgg16_1692
vgg16_1694
vgg16_1696
vgg16_1698
vgg16_1700
vgg16_1702
vgg16_1704
vgg16_1706*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_1348�
flatten/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1714�
 dropout1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout1_layer_call_and_return_conditional_losses_1727�
dense1/StatefulPartitionedCallStatefulPartitionedCall)dropout1/StatefulPartitionedCall:output:0dense1_1740dense1_1742*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_1739�
final/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0
final_1756
final_1758*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_final_layer_call_and_return_conditional_losses_1755u
IdentityIdentity&final/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense1/StatefulPartitionedCall!^dropout1/StatefulPartitionedCall^final/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2D
 dropout1/StatefulPartitionedCall dropout1/StatefulPartitionedCall2>
final/StatefulPartitionedCallfinal/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:$ 

_user_specified_name1758:$ 

_user_specified_name1756:$ 

_user_specified_name1742:$ 

_user_specified_name1740:$ 

_user_specified_name1706:$ 

_user_specified_name1704:$ 

_user_specified_name1702:$ 

_user_specified_name1700:$ 

_user_specified_name1698:$ 

_user_specified_name1696:$ 

_user_specified_name1694:$ 

_user_specified_name1692:$ 

_user_specified_name1690:$ 

_user_specified_name1688:$ 

_user_specified_name1686:$ 

_user_specified_name1684:$ 

_user_specified_name1682:$ 

_user_specified_name1680:$ 

_user_specified_name1678:$ 

_user_specified_name1676:$
 

_user_specified_name1674:$	 

_user_specified_name1672:$ 

_user_specified_name1670:$ 

_user_specified_name1668:$ 

_user_specified_name1666:$ 

_user_specified_name1664:$ 

_user_specified_name1662:$ 

_user_specified_name1660:$ 

_user_specified_name1658:$ 

_user_specified_name1656:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg16_input
�
�
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2305

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������88�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������88�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������88�S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������88�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:���������88�
 
_user_specified_nameinputs
�
�
%__inference_dense1_layer_call_fn_2114

inputs
unknown:���
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_1739p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:�����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2110:$ 

_user_specified_name2108:Q M
)
_output_shapes
:�����������
 
_user_specified_nameinputs
�
�
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2355

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�(
�	
D__inference_VGG16_like_layer_call_and_return_conditional_losses_1836
vgg16_input$

vgg16_1765:@

vgg16_1767:@$

vgg16_1769:@@

vgg16_1771:@%

vgg16_1773:@�

vgg16_1775:	�&

vgg16_1777:��

vgg16_1779:	�&

vgg16_1781:��

vgg16_1783:	�&

vgg16_1785:��

vgg16_1787:	�&

vgg16_1789:��

vgg16_1791:	�&

vgg16_1793:��

vgg16_1795:	�&

vgg16_1797:��

vgg16_1799:	�&

vgg16_1801:��

vgg16_1803:	�&

vgg16_1805:��

vgg16_1807:	�&

vgg16_1809:��

vgg16_1811:	�&

vgg16_1813:��

vgg16_1815:	� 
dense1_1825:���
dense1_1827:	�

final_1830:	�

final_1832:
identity��dense1/StatefulPartitionedCall�final/StatefulPartitionedCall�vgg16/StatefulPartitionedCall�
vgg16/StatefulPartitionedCallStatefulPartitionedCallvgg16_input
vgg16_1765
vgg16_1767
vgg16_1769
vgg16_1771
vgg16_1773
vgg16_1775
vgg16_1777
vgg16_1779
vgg16_1781
vgg16_1783
vgg16_1785
vgg16_1787
vgg16_1789
vgg16_1791
vgg16_1793
vgg16_1795
vgg16_1797
vgg16_1799
vgg16_1801
vgg16_1803
vgg16_1805
vgg16_1807
vgg16_1809
vgg16_1811
vgg16_1813
vgg16_1815*&
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:����������*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_vgg16_layer_call_and_return_conditional_losses_1422�
flatten/PartitionedCallPartitionedCall&vgg16/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_flatten_layer_call_and_return_conditional_losses_1714�
dropout1/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *)
_output_shapes
:�����������* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dropout1_layer_call_and_return_conditional_losses_1823�
dense1/StatefulPartitionedCallStatefulPartitionedCall!dropout1/PartitionedCall:output:0dense1_1825dense1_1827*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense1_layer_call_and_return_conditional_losses_1739�
final/StatefulPartitionedCallStatefulPartitionedCall'dense1/StatefulPartitionedCall:output:0
final_1830
final_1832*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *H
fCRA
?__inference_final_layer_call_and_return_conditional_losses_1755u
IdentityIdentity&final/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^dense1/StatefulPartitionedCall^final/StatefulPartitionedCall^vgg16/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*l
_input_shapes[
Y:�����������: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2@
dense1/StatefulPartitionedCalldense1/StatefulPartitionedCall2>
final/StatefulPartitionedCallfinal/StatefulPartitionedCall2>
vgg16/StatefulPartitionedCallvgg16/StatefulPartitionedCall:$ 

_user_specified_name1832:$ 

_user_specified_name1830:$ 

_user_specified_name1827:$ 

_user_specified_name1825:$ 

_user_specified_name1815:$ 

_user_specified_name1813:$ 

_user_specified_name1811:$ 

_user_specified_name1809:$ 

_user_specified_name1807:$ 

_user_specified_name1805:$ 

_user_specified_name1803:$ 

_user_specified_name1801:$ 

_user_specified_name1799:$ 

_user_specified_name1797:$ 

_user_specified_name1795:$ 

_user_specified_name1793:$ 

_user_specified_name1791:$ 

_user_specified_name1789:$ 

_user_specified_name1787:$ 

_user_specified_name1785:$
 

_user_specified_name1783:$	 

_user_specified_name1781:$ 

_user_specified_name1779:$ 

_user_specified_name1777:$ 

_user_specified_name1775:$ 

_user_specified_name1773:$ 

_user_specified_name1771:$ 

_user_specified_name1769:$ 

_user_specified_name1767:$ 

_user_specified_name1765:^ Z
1
_output_shapes
:�����������
%
_user_specified_namevgg16_input
�
�
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2185

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:�����������@Z
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:�����������@k
IdentityIdentityRelu:activations:0^NoOp*
T0*1
_output_shapes
:�����������@S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
F__inference_block5_conv1_layer_call_and_return_conditional_losses_1308

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2335

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_block1_conv2_layer_call_fn_2174

inputs!
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:�����������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_block1_conv2_layer_call_and_return_conditional_losses_1160y
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:�����������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:�����������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:$ 

_user_specified_name2170:$ 

_user_specified_name2168:Y U
1
_output_shapes
:�����������@
 
_user_specified_nameinputs
�
�
F__inference_block5_conv2_layer_call_and_return_conditional_losses_2425

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:����������Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:����������j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:����������S
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:X T
0
_output_shapes
:����������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
M
vgg16_input>
serving_default_vgg16_input:0�����������9
final0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
layer-10
layer_with_weights-7
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
layer-14
layer_with_weights-10
layer-15
layer_with_weights-11
layer-16
 layer_with_weights-12
 layer-17
!layer-18
"	variables
#trainable_variables
$regularization_losses
%	keras_api
&__call__
*'&call_and_return_all_conditional_losses"
_tf_keras_network
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses"
_tf_keras_layer
�
.	variables
/trainable_variables
0regularization_losses
1	keras_api
2__call__
*3&call_and_return_all_conditional_losses
4_random_generator"
_tf_keras_layer
�
5	variables
6trainable_variables
7regularization_losses
8	keras_api
9__call__
*:&call_and_return_all_conditional_losses

;kernel
<bias"
_tf_keras_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses

Ckernel
Dbias"
_tf_keras_layer
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
W18
X19
Y20
Z21
[22
\23
]24
^25
;26
<27
C28
D29"
trackable_list_wrapper
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
W18
X19
Y20
Z21
[22
\23
]24
^25
;26
<27
C28
D29"
trackable_list_wrapper
 "
trackable_list_wrapper
�
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
dtrace_0
etrace_12�
)__inference_VGG16_like_layer_call_fn_1901
)__inference_VGG16_like_layer_call_fn_1966�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zdtrace_0zetrace_1
�
ftrace_0
gtrace_12�
D__inference_VGG16_like_layer_call_and_return_conditional_losses_1762
D__inference_VGG16_like_layer_call_and_return_conditional_losses_1836�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zftrace_0zgtrace_1
�B�
__inference__wrapped_model_1081vgg16_input"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
h
_variables
i_iterations
j_learning_rate
k_index_dict
l
_momentums
m_velocities
n_update_step_xla"
experimentalOptimizer
,
oserving_default"
signature_map
"
_tf_keras_input_layer
�
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses

Ekernel
Fbias
 v_jit_compiled_convolution_op"
_tf_keras_layer
�
w	variables
xtrainable_variables
yregularization_losses
z	keras_api
{__call__
*|&call_and_return_all_conditional_losses

Gkernel
Hbias
 }_jit_compiled_convolution_op"
_tf_keras_layer
�
~	variables
trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ikernel
Jbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Kkernel
Lbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Mkernel
Nbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Okernel
Pbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Qkernel
Rbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Skernel
Tbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ukernel
Vbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Wkernel
Xbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

Ykernel
Zbias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

[kernel
\bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses

]kernel
^bias
!�_jit_compiled_convolution_op"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
W18
X19
Y20
Z21
[22
\23
]24
^25"
trackable_list_wrapper
�
E0
F1
G2
H3
I4
J5
K6
L7
M8
N9
O10
P11
Q12
R13
S14
T15
U16
V17
W18
X19
Y20
Z21
[22
\23
]24
^25"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
"	variables
#trainable_variables
$regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
$__inference_vgg16_layer_call_fn_1479
$__inference_vgg16_layer_call_fn_1536�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
?__inference_vgg16_layer_call_and_return_conditional_losses_1348
?__inference_vgg16_layer_call_and_return_conditional_losses_1422�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_flatten_layer_call_fn_2072�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_flatten_layer_call_and_return_conditional_losses_2078�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
.	variables
/trainable_variables
0regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
'__inference_dropout1_layer_call_fn_2083
'__inference_dropout1_layer_call_fn_2088�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
B__inference_dropout1_layer_call_and_return_conditional_losses_2100
B__inference_dropout1_layer_call_and_return_conditional_losses_2105�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
"
_generic_user_object
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
5	variables
6trainable_variables
7regularization_losses
9__call__
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_dense1_layer_call_fn_2114�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
@__inference_dense1_layer_call_and_return_conditional_losses_2125�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
": ���2dense1/kernel
:�2dense1/bias
.
C0
D1"
trackable_list_wrapper
.
C0
D1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
$__inference_final_layer_call_fn_2134�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
?__inference_final_layer_call_and_return_conditional_losses_2145�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:	�2final/kernel
:2
final/bias
-:+@2block1_conv1/kernel
:@2block1_conv1/bias
-:+@@2block1_conv2/kernel
:@2block1_conv2/bias
.:,@�2block2_conv1/kernel
 :�2block2_conv1/bias
/:-��2block2_conv2/kernel
 :�2block2_conv2/bias
/:-��2block3_conv1/kernel
 :�2block3_conv1/bias
/:-��2block3_conv2/kernel
 :�2block3_conv2/bias
/:-��2block3_conv3/kernel
 :�2block3_conv3/bias
/:-��2block4_conv1/kernel
 :�2block4_conv1/bias
/:-��2block4_conv2/kernel
 :�2block4_conv2/bias
/:-��2block4_conv3/kernel
 :�2block4_conv3/bias
/:-��2block5_conv1/kernel
 :�2block5_conv1/bias
/:-��2block5_conv2/kernel
 :�2block5_conv2/bias
/:-��2block5_conv3/kernel
 :�2block5_conv3/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_VGG16_like_layer_call_fn_1901vgg16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_VGG16_like_layer_call_fn_1966vgg16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_VGG16_like_layer_call_and_return_conditional_losses_1762vgg16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_VGG16_like_layer_call_and_return_conditional_losses_1836vgg16_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
i0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�54
�55
�56
�57
�58
�59
�60"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29"
trackable_list_wrapper
�
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29"
trackable_list_wrapper
�2��
���
FullArgSpec*
args"�

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
�B�
"__inference_signature_wrapper_2067vgg16_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
  

kwonlyargs�
jvgg16_input
kwonlydefaults
 
annotations� *
 
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block1_conv1_layer_call_fn_2154�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2165�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
w	variables
xtrainable_variables
yregularization_losses
{__call__
*|&call_and_return_all_conditional_losses
&|"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block1_conv2_layer_call_fn_2174�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2185�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
~	variables
trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_block1_pool_layer_call_fn_2190�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_block1_pool_layer_call_and_return_conditional_losses_2195�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
I0
J1"
trackable_list_wrapper
.
I0
J1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block2_conv1_layer_call_fn_2204�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2215�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
K0
L1"
trackable_list_wrapper
.
K0
L1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block2_conv2_layer_call_fn_2224�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2235�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_block2_pool_layer_call_fn_2240�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_block2_pool_layer_call_and_return_conditional_losses_2245�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block3_conv1_layer_call_fn_2254�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2265�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block3_conv2_layer_call_fn_2274�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2285�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block3_conv3_layer_call_fn_2294�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2305�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_block3_pool_layer_call_fn_2310�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_block3_pool_layer_call_and_return_conditional_losses_2315�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block4_conv1_layer_call_fn_2324�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2335�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block4_conv2_layer_call_fn_2344�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2355�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block4_conv3_layer_call_fn_2364�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2375�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_block4_pool_layer_call_fn_2380�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_block4_pool_layer_call_and_return_conditional_losses_2385�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block5_conv1_layer_call_fn_2394�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2405�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block5_conv2_layer_call_fn_2414�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block5_conv2_layer_call_and_return_conditional_losses_2425�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_block5_conv3_layer_call_fn_2434�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_block5_conv3_layer_call_and_return_conditional_losses_2445�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�2��
���
FullArgSpec
args�
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_block5_pool_layer_call_fn_2450�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_block5_pool_layer_call_and_return_conditional_losses_2455�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
 17
!18"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_vgg16_layer_call_fn_1479input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_vgg16_layer_call_fn_1536input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_vgg16_layer_call_and_return_conditional_losses_1348input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_vgg16_layer_call_and_return_conditional_losses_1422input_1"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_flatten_layer_call_fn_2072inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_flatten_layer_call_and_return_conditional_losses_2078inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_dropout1_layer_call_fn_2083inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
'__inference_dropout1_layer_call_fn_2088inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dropout1_layer_call_and_return_conditional_losses_2100inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
B__inference_dropout1_layer_call_and_return_conditional_losses_2105inputs"�
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
%__inference_dense1_layer_call_fn_2114inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
@__inference_dense1_layer_call_and_return_conditional_losses_2125inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_final_layer_call_fn_2134inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_final_layer_call_and_return_conditional_losses_2145inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
R
�	variables
�	keras_api

�total

�count"
_tf_keras_metric
c
�	variables
�	keras_api

�total

�count
�
_fn_kwargs"
_tf_keras_metric
-:+@2m/block1_conv1/kernel
-:+@2v/block1_conv1/kernel
:@2m/block1_conv1/bias
:@2v/block1_conv1/bias
-:+@@2m/block1_conv2/kernel
-:+@@2v/block1_conv2/kernel
:@2m/block1_conv2/bias
:@2v/block1_conv2/bias
.:,@�2m/block2_conv1/kernel
.:,@�2v/block2_conv1/kernel
 :�2m/block2_conv1/bias
 :�2v/block2_conv1/bias
/:-��2m/block2_conv2/kernel
/:-��2v/block2_conv2/kernel
 :�2m/block2_conv2/bias
 :�2v/block2_conv2/bias
/:-��2m/block3_conv1/kernel
/:-��2v/block3_conv1/kernel
 :�2m/block3_conv1/bias
 :�2v/block3_conv1/bias
/:-��2m/block3_conv2/kernel
/:-��2v/block3_conv2/kernel
 :�2m/block3_conv2/bias
 :�2v/block3_conv2/bias
/:-��2m/block3_conv3/kernel
/:-��2v/block3_conv3/kernel
 :�2m/block3_conv3/bias
 :�2v/block3_conv3/bias
/:-��2m/block4_conv1/kernel
/:-��2v/block4_conv1/kernel
 :�2m/block4_conv1/bias
 :�2v/block4_conv1/bias
/:-��2m/block4_conv2/kernel
/:-��2v/block4_conv2/kernel
 :�2m/block4_conv2/bias
 :�2v/block4_conv2/bias
/:-��2m/block4_conv3/kernel
/:-��2v/block4_conv3/kernel
 :�2m/block4_conv3/bias
 :�2v/block4_conv3/bias
/:-��2m/block5_conv1/kernel
/:-��2v/block5_conv1/kernel
 :�2m/block5_conv1/bias
 :�2v/block5_conv1/bias
/:-��2m/block5_conv2/kernel
/:-��2v/block5_conv2/kernel
 :�2m/block5_conv2/bias
 :�2v/block5_conv2/bias
/:-��2m/block5_conv3/kernel
/:-��2v/block5_conv3/kernel
 :�2m/block5_conv3/bias
 :�2v/block5_conv3/bias
": ���2m/dense1/kernel
": ���2v/dense1/kernel
:�2m/dense1/bias
:�2v/dense1/bias
:	�2m/final/kernel
:	�2v/final/kernel
:2m/final/bias
:2v/final/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block1_conv1_layer_call_fn_2154inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2165inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block1_conv2_layer_call_fn_2174inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2185inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_block1_pool_layer_call_fn_2190inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_block1_pool_layer_call_and_return_conditional_losses_2195inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block2_conv1_layer_call_fn_2204inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2215inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block2_conv2_layer_call_fn_2224inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2235inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_block2_pool_layer_call_fn_2240inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_block2_pool_layer_call_and_return_conditional_losses_2245inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block3_conv1_layer_call_fn_2254inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2265inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block3_conv2_layer_call_fn_2274inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2285inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block3_conv3_layer_call_fn_2294inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2305inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_block3_pool_layer_call_fn_2310inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_block3_pool_layer_call_and_return_conditional_losses_2315inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block4_conv1_layer_call_fn_2324inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2335inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block4_conv2_layer_call_fn_2344inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2355inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block4_conv3_layer_call_fn_2364inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2375inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_block4_pool_layer_call_fn_2380inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_block4_pool_layer_call_and_return_conditional_losses_2385inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block5_conv1_layer_call_fn_2394inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2405inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block5_conv2_layer_call_fn_2414inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block5_conv2_layer_call_and_return_conditional_losses_2425inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_block5_conv3_layer_call_fn_2434inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_block5_conv3_layer_call_and_return_conditional_losses_2445inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
*__inference_block5_pool_layer_call_fn_2450inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_block5_pool_layer_call_and_return_conditional_losses_2455inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper�
D__inference_VGG16_like_layer_call_and_return_conditional_losses_1762�EFGHIJKLMNOPQRSTUVWXYZ[\]^;<CDF�C
<�9
/�,
vgg16_input�����������
p

 
� ",�)
"�
tensor_0���������
� �
D__inference_VGG16_like_layer_call_and_return_conditional_losses_1836�EFGHIJKLMNOPQRSTUVWXYZ[\]^;<CDF�C
<�9
/�,
vgg16_input�����������
p 

 
� ",�)
"�
tensor_0���������
� �
)__inference_VGG16_like_layer_call_fn_1901�EFGHIJKLMNOPQRSTUVWXYZ[\]^;<CDF�C
<�9
/�,
vgg16_input�����������
p

 
� "!�
unknown����������
)__inference_VGG16_like_layer_call_fn_1966�EFGHIJKLMNOPQRSTUVWXYZ[\]^;<CDF�C
<�9
/�,
vgg16_input�����������
p 

 
� "!�
unknown����������
__inference__wrapped_model_1081�EFGHIJKLMNOPQRSTUVWXYZ[\]^;<CD>�;
4�1
/�,
vgg16_input�����������
� "-�*
(
final�
final����������
F__inference_block1_conv1_layer_call_and_return_conditional_losses_2165wEF9�6
/�,
*�'
inputs�����������
� "6�3
,�)
tensor_0�����������@
� �
+__inference_block1_conv1_layer_call_fn_2154lEF9�6
/�,
*�'
inputs�����������
� "+�(
unknown�����������@�
F__inference_block1_conv2_layer_call_and_return_conditional_losses_2185wGH9�6
/�,
*�'
inputs�����������@
� "6�3
,�)
tensor_0�����������@
� �
+__inference_block1_conv2_layer_call_fn_2174lGH9�6
/�,
*�'
inputs�����������@
� "+�(
unknown�����������@�
E__inference_block1_pool_layer_call_and_return_conditional_losses_2195�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
*__inference_block1_pool_layer_call_fn_2190�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
F__inference_block2_conv1_layer_call_and_return_conditional_losses_2215tIJ7�4
-�*
(�%
inputs���������pp@
� "5�2
+�(
tensor_0���������pp�
� �
+__inference_block2_conv1_layer_call_fn_2204iIJ7�4
-�*
(�%
inputs���������pp@
� "*�'
unknown���������pp��
F__inference_block2_conv2_layer_call_and_return_conditional_losses_2235uKL8�5
.�+
)�&
inputs���������pp�
� "5�2
+�(
tensor_0���������pp�
� �
+__inference_block2_conv2_layer_call_fn_2224jKL8�5
.�+
)�&
inputs���������pp�
� "*�'
unknown���������pp��
E__inference_block2_pool_layer_call_and_return_conditional_losses_2245�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
*__inference_block2_pool_layer_call_fn_2240�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
F__inference_block3_conv1_layer_call_and_return_conditional_losses_2265uMN8�5
.�+
)�&
inputs���������88�
� "5�2
+�(
tensor_0���������88�
� �
+__inference_block3_conv1_layer_call_fn_2254jMN8�5
.�+
)�&
inputs���������88�
� "*�'
unknown���������88��
F__inference_block3_conv2_layer_call_and_return_conditional_losses_2285uOP8�5
.�+
)�&
inputs���������88�
� "5�2
+�(
tensor_0���������88�
� �
+__inference_block3_conv2_layer_call_fn_2274jOP8�5
.�+
)�&
inputs���������88�
� "*�'
unknown���������88��
F__inference_block3_conv3_layer_call_and_return_conditional_losses_2305uQR8�5
.�+
)�&
inputs���������88�
� "5�2
+�(
tensor_0���������88�
� �
+__inference_block3_conv3_layer_call_fn_2294jQR8�5
.�+
)�&
inputs���������88�
� "*�'
unknown���������88��
E__inference_block3_pool_layer_call_and_return_conditional_losses_2315�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
*__inference_block3_pool_layer_call_fn_2310�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
F__inference_block4_conv1_layer_call_and_return_conditional_losses_2335uST8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
+__inference_block4_conv1_layer_call_fn_2324jST8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
F__inference_block4_conv2_layer_call_and_return_conditional_losses_2355uUV8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
+__inference_block4_conv2_layer_call_fn_2344jUV8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
F__inference_block4_conv3_layer_call_and_return_conditional_losses_2375uWX8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
+__inference_block4_conv3_layer_call_fn_2364jWX8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
E__inference_block4_pool_layer_call_and_return_conditional_losses_2385�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
*__inference_block4_pool_layer_call_fn_2380�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
F__inference_block5_conv1_layer_call_and_return_conditional_losses_2405uYZ8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
+__inference_block5_conv1_layer_call_fn_2394jYZ8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
F__inference_block5_conv2_layer_call_and_return_conditional_losses_2425u[\8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
+__inference_block5_conv2_layer_call_fn_2414j[\8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
F__inference_block5_conv3_layer_call_and_return_conditional_losses_2445u]^8�5
.�+
)�&
inputs����������
� "5�2
+�(
tensor_0����������
� �
+__inference_block5_conv3_layer_call_fn_2434j]^8�5
.�+
)�&
inputs����������
� "*�'
unknown�����������
E__inference_block5_pool_layer_call_and_return_conditional_losses_2455�R�O
H�E
C�@
inputs4������������������������������������
� "O�L
E�B
tensor_04������������������������������������
� �
*__inference_block5_pool_layer_call_fn_2450�R�O
H�E
C�@
inputs4������������������������������������
� "D�A
unknown4�������������������������������������
@__inference_dense1_layer_call_and_return_conditional_losses_2125f;<1�.
'�$
"�
inputs�����������
� "-�*
#� 
tensor_0����������
� �
%__inference_dense1_layer_call_fn_2114[;<1�.
'�$
"�
inputs�����������
� ""�
unknown�����������
B__inference_dropout1_layer_call_and_return_conditional_losses_2100g5�2
+�(
"�
inputs�����������
p
� ".�+
$�!
tensor_0�����������
� �
B__inference_dropout1_layer_call_and_return_conditional_losses_2105g5�2
+�(
"�
inputs�����������
p 
� ".�+
$�!
tensor_0�����������
� �
'__inference_dropout1_layer_call_fn_2083\5�2
+�(
"�
inputs�����������
p
� "#� 
unknown������������
'__inference_dropout1_layer_call_fn_2088\5�2
+�(
"�
inputs�����������
p 
� "#� 
unknown������������
?__inference_final_layer_call_and_return_conditional_losses_2145dCD0�-
&�#
!�
inputs����������
� ",�)
"�
tensor_0���������
� �
$__inference_final_layer_call_fn_2134YCD0�-
&�#
!�
inputs����������
� "!�
unknown����������
A__inference_flatten_layer_call_and_return_conditional_losses_2078j8�5
.�+
)�&
inputs����������
� ".�+
$�!
tensor_0�����������
� �
&__inference_flatten_layer_call_fn_2072_8�5
.�+
)�&
inputs����������
� "#� 
unknown������������
"__inference_signature_wrapper_2067�EFGHIJKLMNOPQRSTUVWXYZ[\]^;<CDM�J
� 
C�@
>
vgg16_input/�,
vgg16_input�����������"-�*
(
final�
final����������
?__inference_vgg16_layer_call_and_return_conditional_losses_1348�EFGHIJKLMNOPQRSTUVWXYZ[\]^B�?
8�5
+�(
input_1�����������
p

 
� "5�2
+�(
tensor_0����������
� �
?__inference_vgg16_layer_call_and_return_conditional_losses_1422�EFGHIJKLMNOPQRSTUVWXYZ[\]^B�?
8�5
+�(
input_1�����������
p 

 
� "5�2
+�(
tensor_0����������
� �
$__inference_vgg16_layer_call_fn_1479�EFGHIJKLMNOPQRSTUVWXYZ[\]^B�?
8�5
+�(
input_1�����������
p

 
� "*�'
unknown�����������
$__inference_vgg16_layer_call_fn_1536�EFGHIJKLMNOPQRSTUVWXYZ[\]^B�?
8�5
+�(
input_1�����������
p 

 
� "*�'
unknown����������