??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
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
dtypetype?
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
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
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
?
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
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.22v2.5.1-97-g957590ea15c8??
{
dense_47/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_47/kernel
t
#dense_47/kernel/Read/ReadVariableOpReadVariableOpdense_47/kernel*
_output_shapes
:	?*
dtype0
s
dense_47/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_47/bias
l
!dense_47/bias/Read/ReadVariableOpReadVariableOpdense_47/bias*
_output_shapes	
:?*
dtype0
{
dense_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
* 
shared_namedense_48/kernel
t
#dense_48/kernel/Read/ReadVariableOpReadVariableOpdense_48/kernel*
_output_shapes
:	?
*
dtype0
r
dense_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_48/bias
k
!dense_48/bias/Read/ReadVariableOpReadVariableOpdense_48/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
?
Adam/dense_47/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_47/kernel/m
?
*Adam/dense_47/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_47/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_47/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_47/bias/m
z
(Adam/dense_47/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_47/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_48/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*'
shared_nameAdam/dense_48/kernel/m
?
*Adam/dense_48/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_48/kernel/m*
_output_shapes
:	?
*
dtype0
?
Adam/dense_48/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_48/bias/m
y
(Adam/dense_48/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_48/bias/m*
_output_shapes
:
*
dtype0
?
Adam/dense_47/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_47/kernel/v
?
*Adam/dense_47/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_47/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_47/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_47/bias/v
z
(Adam/dense_47/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_47/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_48/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?
*'
shared_nameAdam/dense_48/kernel/v
?
*Adam/dense_48/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_48/kernel/v*
_output_shapes
:	?
*
dtype0
?
Adam/dense_48/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/dense_48/bias/v
y
(Adam/dense_48/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_48/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?!
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*? 
value? B?  B? 
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
?
iter

 beta_1

!beta_2
	"decay
#learning_ratemHmImJmKvLvMvNvO

0
1
2
3
 

0
1
2
3
?
$metrics
%layer_metrics
&layer_regularization_losses
trainable_variables

'layers
regularization_losses
	variables
(non_trainable_variables
 
 
 
 
?
)metrics
*layer_metrics
+layer_regularization_losses
trainable_variables

,layers
regularization_losses
	variables
-non_trainable_variables
[Y
VARIABLE_VALUEdense_47/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_47/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
.metrics
/layer_metrics
0layer_regularization_losses
trainable_variables

1layers
regularization_losses
	variables
2non_trainable_variables
 
 
 
?
3metrics
4layer_metrics
5layer_regularization_losses
trainable_variables

6layers
regularization_losses
	variables
7non_trainable_variables
[Y
VARIABLE_VALUEdense_48/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEdense_48/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
8metrics
9layer_metrics
:layer_regularization_losses
trainable_variables

;layers
regularization_losses
	variables
<non_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	?total
	@count
A	variables
B	keras_api
D
	Ctotal
	Dcount
E
_fn_kwargs
F	variables
G	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
@1

A	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

C0
D1

F	variables
~|
VARIABLE_VALUEAdam/dense_47/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_47/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_48/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_48/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_47/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_47/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/dense_48/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUEAdam/dense_48/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
 serving_default_flatten_20_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall serving_default_flatten_20_inputdense_47/kerneldense_47/biasdense_48/kerneldense_48/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_170061
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_47/kernel/Read/ReadVariableOp!dense_47/bias/Read/ReadVariableOp#dense_48/kernel/Read/ReadVariableOp!dense_48/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_47/kernel/m/Read/ReadVariableOp(Adam/dense_47/bias/m/Read/ReadVariableOp*Adam/dense_48/kernel/m/Read/ReadVariableOp(Adam/dense_48/bias/m/Read/ReadVariableOp*Adam/dense_47/kernel/v/Read/ReadVariableOp(Adam/dense_47/bias/v/Read/ReadVariableOp*Adam/dense_48/kernel/v/Read/ReadVariableOp(Adam/dense_48/bias/v/Read/ReadVariableOpConst*"
Tin
2	*
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
GPU 2J 8? *(
f#R!
__inference__traced_save_170300
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_47/kerneldense_47/biasdense_48/kerneldense_48/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1Adam/dense_47/kernel/mAdam/dense_47/bias/mAdam/dense_48/kernel/mAdam/dense_48/bias/mAdam/dense_47/kernel/vAdam/dense_47/bias/vAdam/dense_48/kernel/vAdam/dense_48/bias/v*!
Tin
2*
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
GPU 2J 8? *+
f&R$
"__inference__traced_restore_170373??
?
b
F__inference_flatten_20_layer_call_and_return_conditional_losses_170142

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_21_layer_call_and_return_conditional_losses_169984

inputs"
dense_47_169972:	?
dense_47_169974:	?"
dense_48_169978:	?

dense_48_169980:

identity?? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall?"dropout_22/StatefulPartitionedCall?
flatten_20/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_20_layer_call_and_return_conditional_losses_1698492
flatten_20/PartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0dense_47_169972dense_47_169974*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_1698622"
 dense_47/StatefulPartitionedCall?
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_1699342$
"dropout_22/StatefulPartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_48_169978dense_48_169980*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_1698862"
 dense_48/StatefulPartitionedCall?
IdentityIdentity)dense_48/StatefulPartitionedCall:output:0!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_22_layer_call_and_return_conditional_losses_169934

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_47_layer_call_and_return_conditional_losses_169862

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_21_layer_call_and_return_conditional_losses_170024
flatten_20_input"
dense_47_170012:	?
dense_47_170014:	?"
dense_48_170018:	?

dense_48_170020:

identity?? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall?
flatten_20/PartitionedCallPartitionedCallflatten_20_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_20_layer_call_and_return_conditional_losses_1698492
flatten_20/PartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0dense_47_170012dense_47_170014*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_1698622"
 dense_47/StatefulPartitionedCall?
dropout_22/PartitionedCallPartitionedCall)dense_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_1698732
dropout_22/PartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_48_170018dense_48_170020*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_1698862"
 dense_48/StatefulPartitionedCall?
IdentityIdentity)dense_48/StatefulPartitionedCall:output:0!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_nameflatten_20_input
?
?
.__inference_sequential_21_layer_call_fn_170008
flatten_20_input
unknown:	?
	unknown_0:	?
	unknown_1:	?

	unknown_2:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_20_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_1699842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_nameflatten_20_input
?
d
+__inference_dropout_22_layer_call_fn_170194

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_1699342
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
!__inference__wrapped_model_169836
flatten_20_inputH
5sequential_21_dense_47_matmul_readvariableop_resource:	?E
6sequential_21_dense_47_biasadd_readvariableop_resource:	?H
5sequential_21_dense_48_matmul_readvariableop_resource:	?
D
6sequential_21_dense_48_biasadd_readvariableop_resource:

identity??-sequential_21/dense_47/BiasAdd/ReadVariableOp?,sequential_21/dense_47/MatMul/ReadVariableOp?-sequential_21/dense_48/BiasAdd/ReadVariableOp?,sequential_21/dense_48/MatMul/ReadVariableOp?
sequential_21/flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2 
sequential_21/flatten_20/Const?
 sequential_21/flatten_20/ReshapeReshapeflatten_20_input'sequential_21/flatten_20/Const:output:0*
T0*'
_output_shapes
:?????????2"
 sequential_21/flatten_20/Reshape?
,sequential_21/dense_47/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_47_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,sequential_21/dense_47/MatMul/ReadVariableOp?
sequential_21/dense_47/MatMulMatMul)sequential_21/flatten_20/Reshape:output:04sequential_21/dense_47/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
sequential_21/dense_47/MatMul?
-sequential_21/dense_47/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-sequential_21/dense_47/BiasAdd/ReadVariableOp?
sequential_21/dense_47/BiasAddBiasAdd'sequential_21/dense_47/MatMul:product:05sequential_21/dense_47/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
sequential_21/dense_47/BiasAdd?
sequential_21/dense_47/ReluRelu'sequential_21/dense_47/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
sequential_21/dense_47/Relu?
!sequential_21/dropout_22/IdentityIdentity)sequential_21/dense_47/Relu:activations:0*
T0*(
_output_shapes
:??????????2#
!sequential_21/dropout_22/Identity?
,sequential_21/dense_48/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_48_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02.
,sequential_21/dense_48/MatMul/ReadVariableOp?
sequential_21/dense_48/MatMulMatMul*sequential_21/dropout_22/Identity:output:04sequential_21/dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
sequential_21/dense_48/MatMul?
-sequential_21/dense_48/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_48_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02/
-sequential_21/dense_48/BiasAdd/ReadVariableOp?
sequential_21/dense_48/BiasAddBiasAdd'sequential_21/dense_48/MatMul:product:05sequential_21/dense_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2 
sequential_21/dense_48/BiasAdd?
sequential_21/dense_48/SoftmaxSoftmax'sequential_21/dense_48/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2 
sequential_21/dense_48/Softmax?
IdentityIdentity(sequential_21/dense_48/Softmax:softmax:0.^sequential_21/dense_47/BiasAdd/ReadVariableOp-^sequential_21/dense_47/MatMul/ReadVariableOp.^sequential_21/dense_48/BiasAdd/ReadVariableOp-^sequential_21/dense_48/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2^
-sequential_21/dense_47/BiasAdd/ReadVariableOp-sequential_21/dense_47/BiasAdd/ReadVariableOp2\
,sequential_21/dense_47/MatMul/ReadVariableOp,sequential_21/dense_47/MatMul/ReadVariableOp2^
-sequential_21/dense_48/BiasAdd/ReadVariableOp-sequential_21/dense_48/BiasAdd/ReadVariableOp2\
,sequential_21/dense_48/MatMul/ReadVariableOp,sequential_21/dense_48/MatMul/ReadVariableOp:Y U
'
_output_shapes
:?????????
*
_user_specified_nameflatten_20_input
?
?
.__inference_sequential_21_layer_call_fn_170136

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?

	unknown_2:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_1699842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_21_layer_call_fn_169904
flatten_20_input
unknown:	?
	unknown_0:	?
	unknown_1:	?

	unknown_2:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_20_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_1698932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_nameflatten_20_input
?
b
F__inference_flatten_20_layer_call_and_return_conditional_losses_169849

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Constg
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:?????????2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_dense_47_layer_call_fn_170167

inputs
unknown:	?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_1698622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_21_layer_call_and_return_conditional_losses_170082

inputs:
'dense_47_matmul_readvariableop_resource:	?7
(dense_47_biasadd_readvariableop_resource:	?:
'dense_48_matmul_readvariableop_resource:	?
6
(dense_48_biasadd_readvariableop_resource:

identity??dense_47/BiasAdd/ReadVariableOp?dense_47/MatMul/ReadVariableOp?dense_48/BiasAdd/ReadVariableOp?dense_48/MatMul/ReadVariableOpu
flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_20/Const?
flatten_20/ReshapeReshapeinputsflatten_20/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_20/Reshape?
dense_47/MatMul/ReadVariableOpReadVariableOp'dense_47_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_47/MatMul/ReadVariableOp?
dense_47/MatMulMatMulflatten_20/Reshape:output:0&dense_47/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_47/MatMul?
dense_47/BiasAdd/ReadVariableOpReadVariableOp(dense_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_47/BiasAdd/ReadVariableOp?
dense_47/BiasAddBiasAdddense_47/MatMul:product:0'dense_47/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_47/BiasAddt
dense_47/ReluReludense_47/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_47/Relu?
dropout_22/IdentityIdentitydense_47/Relu:activations:0*
T0*(
_output_shapes
:??????????2
dropout_22/Identity?
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_48/MatMul/ReadVariableOp?
dense_48/MatMulMatMuldropout_22/Identity:output:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_48/MatMul?
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_48/BiasAdd/ReadVariableOp?
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_48/BiasAdd|
dense_48/SoftmaxSoftmaxdense_48/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_48/Softmax?
IdentityIdentitydense_48/Softmax:softmax:0 ^dense_47/BiasAdd/ReadVariableOp^dense_47/MatMul/ReadVariableOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2B
dense_47/BiasAdd/ReadVariableOpdense_47/BiasAdd/ReadVariableOp2@
dense_47/MatMul/ReadVariableOpdense_47/MatMul/ReadVariableOp2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
e
F__inference_dropout_22_layer_call_and_return_conditional_losses_170184

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_47_layer_call_and_return_conditional_losses_170158

inputs1
matmul_readvariableop_resource:	?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
I__inference_sequential_21_layer_call_and_return_conditional_losses_170040
flatten_20_input"
dense_47_170028:	?
dense_47_170030:	?"
dense_48_170034:	?

dense_48_170036:

identity?? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall?"dropout_22/StatefulPartitionedCall?
flatten_20/PartitionedCallPartitionedCallflatten_20_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_20_layer_call_and_return_conditional_losses_1698492
flatten_20/PartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0dense_47_170028dense_47_170030*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_1698622"
 dense_47/StatefulPartitionedCall?
"dropout_22/StatefulPartitionedCallStatefulPartitionedCall)dense_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_1699342$
"dropout_22/StatefulPartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall+dropout_22/StatefulPartitionedCall:output:0dense_48_170034dense_48_170036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_1698862"
 dense_48/StatefulPartitionedCall?
IdentityIdentity)dense_48/StatefulPartitionedCall:output:0!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall#^dropout_22/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall2H
"dropout_22/StatefulPartitionedCall"dropout_22/StatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_nameflatten_20_input
?
G
+__inference_dropout_22_layer_call_fn_170189

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_1698732
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_170172

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?[
?
"__inference__traced_restore_170373
file_prefix3
 assignvariableop_dense_47_kernel:	?/
 assignvariableop_1_dense_47_bias:	?5
"assignvariableop_2_dense_48_kernel:	?
.
 assignvariableop_3_dense_48_bias:
&
assignvariableop_4_adam_iter:	 (
assignvariableop_5_adam_beta_1: (
assignvariableop_6_adam_beta_2: '
assignvariableop_7_adam_decay: /
%assignvariableop_8_adam_learning_rate: "
assignvariableop_9_total: #
assignvariableop_10_count: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: =
*assignvariableop_13_adam_dense_47_kernel_m:	?7
(assignvariableop_14_adam_dense_47_bias_m:	?=
*assignvariableop_15_adam_dense_48_kernel_m:	?
6
(assignvariableop_16_adam_dense_48_bias_m:
=
*assignvariableop_17_adam_dense_47_kernel_v:	?7
(assignvariableop_18_adam_dense_47_bias_v:	?=
*assignvariableop_19_adam_dense_48_kernel_v:	?
6
(assignvariableop_20_adam_dense_48_bias_v:

identity_22??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp assignvariableop_dense_47_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_47_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_48_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_48_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_iterIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_beta_1Identity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_2Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_decayIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp%assignvariableop_8_adam_learning_rateIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_totalIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_countIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp*assignvariableop_13_adam_dense_47_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp(assignvariableop_14_adam_dense_47_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_adam_dense_48_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp(assignvariableop_16_adam_dense_48_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp*assignvariableop_17_adam_dense_47_kernel_vIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_adam_dense_47_bias_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp*assignvariableop_19_adam_dense_48_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp(assignvariableop_20_adam_dense_48_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_209
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_21?
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_22"#
identity_22Identity_22:output:0*?
_input_shapes.
,: : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
I__inference_sequential_21_layer_call_and_return_conditional_losses_169893

inputs"
dense_47_169863:	?
dense_47_169865:	?"
dense_48_169887:	?

dense_48_169889:

identity?? dense_47/StatefulPartitionedCall? dense_48/StatefulPartitionedCall?
flatten_20/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_20_layer_call_and_return_conditional_losses_1698492
flatten_20/PartitionedCall?
 dense_47/StatefulPartitionedCallStatefulPartitionedCall#flatten_20/PartitionedCall:output:0dense_47_169863dense_47_169865*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_47_layer_call_and_return_conditional_losses_1698622"
 dense_47/StatefulPartitionedCall?
dropout_22/PartitionedCallPartitionedCall)dense_47/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_dropout_22_layer_call_and_return_conditional_losses_1698732
dropout_22/PartitionedCall?
 dense_48/StatefulPartitionedCallStatefulPartitionedCall#dropout_22/PartitionedCall:output:0dense_48_169887dense_48_169889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_1698862"
 dense_48/StatefulPartitionedCall?
IdentityIdentity)dense_48/StatefulPartitionedCall:output:0!^dense_47/StatefulPartitionedCall!^dense_48/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2D
 dense_47/StatefulPartitionedCall dense_47/StatefulPartitionedCall2D
 dense_48/StatefulPartitionedCall dense_48/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_dense_48_layer_call_fn_170214

inputs
unknown:	?

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *M
fHRF
D__inference_dense_48_layer_call_and_return_conditional_losses_1698862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
D__inference_dense_48_layer_call_and_return_conditional_losses_169886

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
$__inference_signature_wrapper_170061
flatten_20_input
unknown:	?
	unknown_0:	?
	unknown_1:	?

	unknown_2:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallflatten_20_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_1698362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
'
_output_shapes
:?????????
*
_user_specified_nameflatten_20_input
?
d
F__inference_dropout_22_layer_call_and_return_conditional_losses_169873

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:??????????2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:??????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
.__inference_sequential_21_layer_call_fn_170123

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?

	unknown_2:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_1698932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?3
?
__inference__traced_save_170300
file_prefix.
*savev2_dense_47_kernel_read_readvariableop,
(savev2_dense_47_bias_read_readvariableop.
*savev2_dense_48_kernel_read_readvariableop,
(savev2_dense_48_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_47_kernel_m_read_readvariableop3
/savev2_adam_dense_47_bias_m_read_readvariableop5
1savev2_adam_dense_48_kernel_m_read_readvariableop3
/savev2_adam_dense_48_bias_m_read_readvariableop5
1savev2_adam_dense_47_kernel_v_read_readvariableop3
/savev2_adam_dense_47_bias_v_read_readvariableop5
1savev2_adam_dense_48_kernel_v_read_readvariableop3
/savev2_adam_dense_48_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?

value?
B?
B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_47_kernel_read_readvariableop(savev2_dense_47_bias_read_readvariableop*savev2_dense_48_kernel_read_readvariableop(savev2_dense_48_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_47_kernel_m_read_readvariableop/savev2_adam_dense_47_bias_m_read_readvariableop1savev2_adam_dense_48_kernel_m_read_readvariableop/savev2_adam_dense_48_bias_m_read_readvariableop1savev2_adam_dense_47_kernel_v_read_readvariableop/savev2_adam_dense_47_bias_v_read_readvariableop1savev2_adam_dense_48_kernel_v_read_readvariableop/savev2_adam_dense_48_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *$
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
: :	?:?:	?
:
: : : : : : : : : :	?:?:	?
:
:	?:?:	?
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:%!

_output_shapes
:	?:!

_output_shapes	
:?:%!

_output_shapes
:	?
: 

_output_shapes
:
:

_output_shapes
: 
?
G
+__inference_flatten_20_layer_call_fn_170147

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_flatten_20_layer_call_and_return_conditional_losses_1698492
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?"
?
I__inference_sequential_21_layer_call_and_return_conditional_losses_170110

inputs:
'dense_47_matmul_readvariableop_resource:	?7
(dense_47_biasadd_readvariableop_resource:	?:
'dense_48_matmul_readvariableop_resource:	?
6
(dense_48_biasadd_readvariableop_resource:

identity??dense_47/BiasAdd/ReadVariableOp?dense_47/MatMul/ReadVariableOp?dense_48/BiasAdd/ReadVariableOp?dense_48/MatMul/ReadVariableOpu
flatten_20/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten_20/Const?
flatten_20/ReshapeReshapeinputsflatten_20/Const:output:0*
T0*'
_output_shapes
:?????????2
flatten_20/Reshape?
dense_47/MatMul/ReadVariableOpReadVariableOp'dense_47_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02 
dense_47/MatMul/ReadVariableOp?
dense_47/MatMulMatMulflatten_20/Reshape:output:0&dense_47/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_47/MatMul?
dense_47/BiasAdd/ReadVariableOpReadVariableOp(dense_47_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
dense_47/BiasAdd/ReadVariableOp?
dense_47/BiasAddBiasAdddense_47/MatMul:product:0'dense_47/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_47/BiasAddt
dense_47/ReluReludense_47/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_47/Reluy
dropout_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout_22/dropout/Const?
dropout_22/dropout/MulMuldense_47/Relu:activations:0!dropout_22/dropout/Const:output:0*
T0*(
_output_shapes
:??????????2
dropout_22/dropout/Mul
dropout_22/dropout/ShapeShapedense_47/Relu:activations:0*
T0*
_output_shapes
:2
dropout_22/dropout/Shape?
/dropout_22/dropout/random_uniform/RandomUniformRandomUniform!dropout_22/dropout/Shape:output:0*
T0*(
_output_shapes
:??????????*
dtype021
/dropout_22/dropout/random_uniform/RandomUniform?
!dropout_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2#
!dropout_22/dropout/GreaterEqual/y?
dropout_22/dropout/GreaterEqualGreaterEqual8dropout_22/dropout/random_uniform/RandomUniform:output:0*dropout_22/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:??????????2!
dropout_22/dropout/GreaterEqual?
dropout_22/dropout/CastCast#dropout_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:??????????2
dropout_22/dropout/Cast?
dropout_22/dropout/Mul_1Muldropout_22/dropout/Mul:z:0dropout_22/dropout/Cast:y:0*
T0*(
_output_shapes
:??????????2
dropout_22/dropout/Mul_1?
dense_48/MatMul/ReadVariableOpReadVariableOp'dense_48_matmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02 
dense_48/MatMul/ReadVariableOp?
dense_48/MatMulMatMuldropout_22/dropout/Mul_1:z:0&dense_48/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_48/MatMul?
dense_48/BiasAdd/ReadVariableOpReadVariableOp(dense_48_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02!
dense_48/BiasAdd/ReadVariableOp?
dense_48/BiasAddBiasAdddense_48/MatMul:product:0'dense_48/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
dense_48/BiasAdd|
dense_48/SoftmaxSoftmaxdense_48/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
2
dense_48/Softmax?
IdentityIdentitydense_48/Softmax:softmax:0 ^dense_47/BiasAdd/ReadVariableOp^dense_47/MatMul/ReadVariableOp ^dense_48/BiasAdd/ReadVariableOp^dense_48/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : : : 2B
dense_47/BiasAdd/ReadVariableOpdense_47/BiasAdd/ReadVariableOp2@
dense_47/MatMul/ReadVariableOpdense_47/MatMul/ReadVariableOp2B
dense_48/BiasAdd/ReadVariableOpdense_48/BiasAdd/ReadVariableOp2@
dense_48/MatMul/ReadVariableOpdense_48/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_dense_48_layer_call_and_return_conditional_losses_170205

inputs1
matmul_readvariableop_resource:	?
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
M
flatten_20_input9
"serving_default_flatten_20_input:0?????????<
dense_480
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:??
?#
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
*P&call_and_return_all_conditional_losses
Q__call__
R_default_save_signature"?!
_tf_keras_sequential? {"name": "sequential_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_21", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_20_input"}}, {"class_name": "Flatten", "config": {"name": "flatten_20", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_47", "trainable": true, "dtype": "float32", "units": 655, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_48", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 28]}, "float32", "flatten_20_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_21", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 28]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "flatten_20_input"}, "shared_object_id": 0}, {"class_name": "Flatten", "config": {"name": "flatten_20", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 1}, {"class_name": "Dense", "config": {"name": "dense_47", "trainable": true, "dtype": "float32", "units": 655, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}, {"class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "shared_object_id": 5}, {"class_name": "Dense", "config": {"name": "dense_48", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 11}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*S&call_and_return_all_conditional_losses
T__call__"?
_tf_keras_layer?{"name": "flatten_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_20", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 1, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 10}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*U&call_and_return_all_conditional_losses
V__call__"?
_tf_keras_layer?{"name": "dense_47", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_47", "trainable": true, "dtype": "float32", "units": 655, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 28}}, "shared_object_id": 12}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*W&call_and_return_all_conditional_losses
X__call__"?
_tf_keras_layer?{"name": "dropout_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_22", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}, "shared_object_id": 5}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*Y&call_and_return_all_conditional_losses
Z__call__"?
_tf_keras_layer?{"name": "dense_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_48", "trainable": true, "dtype": "float32", "units": 10, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 6}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 7}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 655}}, "shared_object_id": 13}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 655]}}
?
iter

 beta_1

!beta_2
	"decay
#learning_ratemHmImJmKvLvMvNvO"
	optimizer
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
?
$metrics
%layer_metrics
&layer_regularization_losses
trainable_variables

'layers
regularization_losses
	variables
(non_trainable_variables
Q__call__
R_default_save_signature
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
,
[serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
)metrics
*layer_metrics
+layer_regularization_losses
trainable_variables

,layers
regularization_losses
	variables
-non_trainable_variables
T__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
": 	?2dense_47/kernel
:?2dense_47/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
.metrics
/layer_metrics
0layer_regularization_losses
trainable_variables

1layers
regularization_losses
	variables
2non_trainable_variables
V__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
3metrics
4layer_metrics
5layer_regularization_losses
trainable_variables

6layers
regularization_losses
	variables
7non_trainable_variables
X__call__
*W&call_and_return_all_conditional_losses
&W"call_and_return_conditional_losses"
_generic_user_object
": 	?
2dense_48/kernel
:
2dense_48/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
8metrics
9layer_metrics
:layer_regularization_losses
trainable_variables

;layers
regularization_losses
	variables
<non_trainable_variables
Z__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
.
=0
>1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	?total
	@count
A	variables
B	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 14}
?
	Ctotal
	Dcount
E
_fn_kwargs
F	variables
G	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 11}
:  (2total
:  (2count
.
?0
@1"
trackable_list_wrapper
-
A	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
C0
D1"
trackable_list_wrapper
-
F	variables"
_generic_user_object
':%	?2Adam/dense_47/kernel/m
!:?2Adam/dense_47/bias/m
':%	?
2Adam/dense_48/kernel/m
 :
2Adam/dense_48/bias/m
':%	?2Adam/dense_47/kernel/v
!:?2Adam/dense_47/bias/v
':%	?
2Adam/dense_48/kernel/v
 :
2Adam/dense_48/bias/v
?2?
I__inference_sequential_21_layer_call_and_return_conditional_losses_170082
I__inference_sequential_21_layer_call_and_return_conditional_losses_170110
I__inference_sequential_21_layer_call_and_return_conditional_losses_170024
I__inference_sequential_21_layer_call_and_return_conditional_losses_170040?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_sequential_21_layer_call_fn_169904
.__inference_sequential_21_layer_call_fn_170123
.__inference_sequential_21_layer_call_fn_170136
.__inference_sequential_21_layer_call_fn_170008?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_169836?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
*?'
flatten_20_input?????????
?2?
F__inference_flatten_20_layer_call_and_return_conditional_losses_170142?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_flatten_20_layer_call_fn_170147?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_47_layer_call_and_return_conditional_losses_170158?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_47_layer_call_fn_170167?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_dropout_22_layer_call_and_return_conditional_losses_170172
F__inference_dropout_22_layer_call_and_return_conditional_losses_170184?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
+__inference_dropout_22_layer_call_fn_170189
+__inference_dropout_22_layer_call_fn_170194?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
D__inference_dense_48_layer_call_and_return_conditional_losses_170205?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_dense_48_layer_call_fn_170214?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
$__inference_signature_wrapper_170061flatten_20_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
!__inference__wrapped_model_169836v9?6
/?,
*?'
flatten_20_input?????????
? "3?0
.
dense_48"?
dense_48?????????
?
D__inference_dense_47_layer_call_and_return_conditional_losses_170158]/?,
%?"
 ?
inputs?????????
? "&?#
?
0??????????
? }
)__inference_dense_47_layer_call_fn_170167P/?,
%?"
 ?
inputs?????????
? "????????????
D__inference_dense_48_layer_call_and_return_conditional_losses_170205]0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????

? }
)__inference_dense_48_layer_call_fn_170214P0?-
&?#
!?
inputs??????????
? "??????????
?
F__inference_dropout_22_layer_call_and_return_conditional_losses_170172^4?1
*?'
!?
inputs??????????
p 
? "&?#
?
0??????????
? ?
F__inference_dropout_22_layer_call_and_return_conditional_losses_170184^4?1
*?'
!?
inputs??????????
p
? "&?#
?
0??????????
? ?
+__inference_dropout_22_layer_call_fn_170189Q4?1
*?'
!?
inputs??????????
p 
? "????????????
+__inference_dropout_22_layer_call_fn_170194Q4?1
*?'
!?
inputs??????????
p
? "????????????
F__inference_flatten_20_layer_call_and_return_conditional_losses_170142X/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
+__inference_flatten_20_layer_call_fn_170147K/?,
%?"
 ?
inputs?????????
? "???????????
I__inference_sequential_21_layer_call_and_return_conditional_losses_170024pA?>
7?4
*?'
flatten_20_input?????????
p 

 
? "%?"
?
0?????????

? ?
I__inference_sequential_21_layer_call_and_return_conditional_losses_170040pA?>
7?4
*?'
flatten_20_input?????????
p

 
? "%?"
?
0?????????

? ?
I__inference_sequential_21_layer_call_and_return_conditional_losses_170082f7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????

? ?
I__inference_sequential_21_layer_call_and_return_conditional_losses_170110f7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????

? ?
.__inference_sequential_21_layer_call_fn_169904cA?>
7?4
*?'
flatten_20_input?????????
p 

 
? "??????????
?
.__inference_sequential_21_layer_call_fn_170008cA?>
7?4
*?'
flatten_20_input?????????
p

 
? "??????????
?
.__inference_sequential_21_layer_call_fn_170123Y7?4
-?*
 ?
inputs?????????
p 

 
? "??????????
?
.__inference_sequential_21_layer_call_fn_170136Y7?4
-?*
 ?
inputs?????????
p

 
? "??????????
?
$__inference_signature_wrapper_170061?M?J
? 
C?@
>
flatten_20_input*?'
flatten_20_input?????????"3?0
.
dense_48"?
dense_48?????????
