
«
B
AddV2
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype
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
delete_old_dirsbool(
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
dtypetype
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
¾
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
executor_typestring 
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.4.12v2.4.0-49-g85c8b2a817f8ø½
¨
&my_sequential_model/flexible_dense_1/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&my_sequential_model/flexible_dense_1/w
¡
:my_sequential_model/flexible_dense_1/w/Read/ReadVariableOpReadVariableOp&my_sequential_model/flexible_dense_1/w*
_output_shapes

:*
dtype0
¤
&my_sequential_model/flexible_dense_1/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&my_sequential_model/flexible_dense_1/b

:my_sequential_model/flexible_dense_1/b/Read/ReadVariableOpReadVariableOp&my_sequential_model/flexible_dense_1/b*
_output_shapes
:*
dtype0
¨
&my_sequential_model/flexible_dense_2/wVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&my_sequential_model/flexible_dense_2/w
¡
:my_sequential_model/flexible_dense_2/w/Read/ReadVariableOpReadVariableOp&my_sequential_model/flexible_dense_2/w*
_output_shapes

:*
dtype0
¤
&my_sequential_model/flexible_dense_2/bVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&my_sequential_model/flexible_dense_2/b

:my_sequential_model/flexible_dense_2/b/Read/ReadVariableOpReadVariableOp&my_sequential_model/flexible_dense_2/b*
_output_shapes
:*
dtype0

NoOpNoOp
ã
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB B
|
dense_1
dense_2
	variables
regularization_losses
trainable_variables
	keras_api

signatures
`
w
	b

trainable_variables
	variables
regularization_losses
	keras_api
`
w
b
trainable_variables
	variables
regularization_losses
	keras_api

0
	1
2
3
 

0
	1
2
3
­
	variables
regularization_losses
non_trainable_variables
layer_metrics
layer_regularization_losses
metrics

layers
trainable_variables
 
`^
VARIABLE_VALUE&my_sequential_model/flexible_dense_1/w$dense_1/w/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE&my_sequential_model/flexible_dense_1/b$dense_1/b/.ATTRIBUTES/VARIABLE_VALUE

0
	1

0
	1
 
­

trainable_variables
	variables
layer_metrics
non_trainable_variables
layer_regularization_losses
metrics

layers
regularization_losses
`^
VARIABLE_VALUE&my_sequential_model/flexible_dense_2/w$dense_2/w/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUE&my_sequential_model/flexible_dense_2/b$dense_2/b/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
­
trainable_variables
	variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
!metrics

"layers
regularization_losses
 
 
 
 

0
1
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
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
Ú
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1&my_sequential_model/flexible_dense_1/w&my_sequential_model/flexible_dense_1/b&my_sequential_model/flexible_dense_2/w&my_sequential_model/flexible_dense_2/b*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference_signature_wrapper_883
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename:my_sequential_model/flexible_dense_1/w/Read/ReadVariableOp:my_sequential_model/flexible_dense_1/b/Read/ReadVariableOp:my_sequential_model/flexible_dense_2/w/Read/ReadVariableOp:my_sequential_model/flexible_dense_2/b/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *%
f R
__inference__traced_save_956
º
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename&my_sequential_model/flexible_dense_1/w&my_sequential_model/flexible_dense_1/b&my_sequential_model/flexible_dense_2/w&my_sequential_model/flexible_dense_2/b*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_restore_978Ô
ß
Ú
I__inference_flexible_dense_1_layer_call_and_return_conditional_losses_811

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity¢MatMul/ReadVariableOp¢add/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
IdentityIdentityadd:z:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

.__inference_flexible_dense_2_layer_call_fn_921

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_8372
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ

!__inference_signature_wrapper_883
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *'
f"R 
__inference__wrapped_model_7972
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ß
Ú
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_912

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity¢MatMul/ReadVariableOp¢add/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
IdentityIdentityadd:z:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾

__inference__traced_restore_978
file_prefix;
7assignvariableop_my_sequential_model_flexible_dense_1_w=
9assignvariableop_1_my_sequential_model_flexible_dense_1_b=
9assignvariableop_2_my_sequential_model_flexible_dense_2_w=
9assignvariableop_3_my_sequential_model_flexible_dense_2_b

identity_5¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_2¢AssignVariableOp_3¿
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ë
valueÁB¾B$dense_1/w/.ATTRIBUTES/VARIABLE_VALUEB$dense_1/b/.ATTRIBUTES/VARIABLE_VALUEB$dense_2/w/.ATTRIBUTES/VARIABLE_VALUEB$dense_2/b/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slicesÄ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity¶
AssignVariableOpAssignVariableOp7assignvariableop_my_sequential_model_flexible_dense_1_wIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¾
AssignVariableOp_1AssignVariableOp9assignvariableop_1_my_sequential_model_flexible_dense_1_bIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2¾
AssignVariableOp_2AssignVariableOp9assignvariableop_2_my_sequential_model_flexible_dense_2_wIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¾
AssignVariableOp_3AssignVariableOp9assignvariableop_3_my_sequential_model_flexible_dense_2_bIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpº

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4¬

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
³
¥
1__inference_my_sequential_model_layer_call_fn_868
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *U
fPRN
L__inference_my_sequential_model_layer_call_and_return_conditional_losses_8542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
þ
ª
L__inference_my_sequential_model_layer_call_and_return_conditional_losses_854
input_1
flexible_dense_1_822
flexible_dense_1_824
flexible_dense_2_848
flexible_dense_2_850
identity¢(flexible_dense_1/StatefulPartitionedCall¢(flexible_dense_2/StatefulPartitionedCall·
(flexible_dense_1/StatefulPartitionedCallStatefulPartitionedCallinput_1flexible_dense_1_822flexible_dense_1_824*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_flexible_dense_1_layer_call_and_return_conditional_losses_8112*
(flexible_dense_1/StatefulPartitionedCallá
(flexible_dense_2/StatefulPartitionedCallStatefulPartitionedCall1flexible_dense_1/StatefulPartitionedCall:output:0flexible_dense_2_848flexible_dense_2_850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_8372*
(flexible_dense_2/StatefulPartitionedCallÛ
IdentityIdentity1flexible_dense_2/StatefulPartitionedCall:output:0)^flexible_dense_1/StatefulPartitionedCall)^flexible_dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2T
(flexible_dense_1/StatefulPartitionedCall(flexible_dense_1/StatefulPartitionedCall2T
(flexible_dense_2/StatefulPartitionedCall(flexible_dense_2/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1
ê

.__inference_flexible_dense_1_layer_call_fn_902

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallü
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_flexible_dense_1_layer_call_and_return_conditional_losses_8112
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ò

__inference__traced_save_956
file_prefixE
Asavev2_my_sequential_model_flexible_dense_1_w_read_readvariableopE
Asavev2_my_sequential_model_flexible_dense_1_b_read_readvariableopE
Asavev2_my_sequential_model_flexible_dense_2_w_read_readvariableopE
Asavev2_my_sequential_model_flexible_dense_2_b_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
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
Const_1
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
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¹
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*Ë
valueÁB¾B$dense_1/w/.ATTRIBUTES/VARIABLE_VALUEB$dense_1/b/.ATTRIBUTES/VARIABLE_VALUEB$dense_2/w/.ATTRIBUTES/VARIABLE_VALUEB$dense_2/b/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slicesÊ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Asavev2_my_sequential_model_flexible_dense_1_w_read_readvariableopAsavev2_my_sequential_model_flexible_dense_1_b_read_readvariableopAsavev2_my_sequential_model_flexible_dense_2_w_read_readvariableopAsavev2_my_sequential_model_flexible_dense_2_b_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
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

identity_1Identity_1:output:0*7
_input_shapes&
$: ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
ß
Ú
I__inference_flexible_dense_1_layer_call_and_return_conditional_losses_893

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity¢MatMul/ReadVariableOp¢add/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
IdentityIdentityadd:z:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
Ú
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_837

inputs"
matmul_readvariableop_resource
add_readvariableop_resource
identity¢MatMul/ReadVariableOp¢add/ReadVariableOp
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
:*
dtype02
add/ReadVariableOps
addAddV2MatMul:product:0add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
add
IdentityIdentityadd:z:0^MatMul/ReadVariableOp^add/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
add/ReadVariableOpadd/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ª
Ê
__inference__wrapped_model_797
input_1G
Cmy_sequential_model_flexible_dense_1_matmul_readvariableop_resourceD
@my_sequential_model_flexible_dense_1_add_readvariableop_resourceG
Cmy_sequential_model_flexible_dense_2_matmul_readvariableop_resourceD
@my_sequential_model_flexible_dense_2_add_readvariableop_resource
identity¢:my_sequential_model/flexible_dense_1/MatMul/ReadVariableOp¢7my_sequential_model/flexible_dense_1/add/ReadVariableOp¢:my_sequential_model/flexible_dense_2/MatMul/ReadVariableOp¢7my_sequential_model/flexible_dense_2/add/ReadVariableOpü
:my_sequential_model/flexible_dense_1/MatMul/ReadVariableOpReadVariableOpCmy_sequential_model_flexible_dense_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:my_sequential_model/flexible_dense_1/MatMul/ReadVariableOpã
+my_sequential_model/flexible_dense_1/MatMulMatMulinput_1Bmy_sequential_model/flexible_dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+my_sequential_model/flexible_dense_1/MatMulï
7my_sequential_model/flexible_dense_1/add/ReadVariableOpReadVariableOp@my_sequential_model_flexible_dense_1_add_readvariableop_resource*
_output_shapes
:*
dtype029
7my_sequential_model/flexible_dense_1/add/ReadVariableOp
(my_sequential_model/flexible_dense_1/addAddV25my_sequential_model/flexible_dense_1/MatMul:product:0?my_sequential_model/flexible_dense_1/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(my_sequential_model/flexible_dense_1/addü
:my_sequential_model/flexible_dense_2/MatMul/ReadVariableOpReadVariableOpCmy_sequential_model_flexible_dense_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype02<
:my_sequential_model/flexible_dense_2/MatMul/ReadVariableOp
+my_sequential_model/flexible_dense_2/MatMulMatMul,my_sequential_model/flexible_dense_1/add:z:0Bmy_sequential_model/flexible_dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+my_sequential_model/flexible_dense_2/MatMulï
7my_sequential_model/flexible_dense_2/add/ReadVariableOpReadVariableOp@my_sequential_model_flexible_dense_2_add_readvariableop_resource*
_output_shapes
:*
dtype029
7my_sequential_model/flexible_dense_2/add/ReadVariableOp
(my_sequential_model/flexible_dense_2/addAddV25my_sequential_model/flexible_dense_2/MatMul:product:0?my_sequential_model/flexible_dense_2/add/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(my_sequential_model/flexible_dense_2/addî
IdentityIdentity,my_sequential_model/flexible_dense_2/add:z:0;^my_sequential_model/flexible_dense_1/MatMul/ReadVariableOp8^my_sequential_model/flexible_dense_1/add/ReadVariableOp;^my_sequential_model/flexible_dense_2/MatMul/ReadVariableOp8^my_sequential_model/flexible_dense_2/add/ReadVariableOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ::::2x
:my_sequential_model/flexible_dense_1/MatMul/ReadVariableOp:my_sequential_model/flexible_dense_1/MatMul/ReadVariableOp2r
7my_sequential_model/flexible_dense_1/add/ReadVariableOp7my_sequential_model/flexible_dense_1/add/ReadVariableOp2x
:my_sequential_model/flexible_dense_2/MatMul/ReadVariableOp:my_sequential_model/flexible_dense_2/MatMul/ReadVariableOp2r
7my_sequential_model/flexible_dense_2/add/ReadVariableOp7my_sequential_model/flexible_dense_2/add/ReadVariableOp:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
serving_default
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ<
output_10
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:6
í
dense_1
dense_2
	variables
regularization_losses
trainable_variables
	keras_api

signatures
#__call__
*$&call_and_return_all_conditional_losses
%_default_save_signature"
_tf_keras_modelý{"class_name": "MySequentialModel", "name": "my_sequential_model", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MySequentialModel"}}

w
	b

trainable_variables
	variables
regularization_losses
	keras_api
&__call__
*'&call_and_return_all_conditional_losses"å
_tf_keras_layerË{"class_name": "FlexibleDense", "name": "flexible_dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 3]}}

w
b
trainable_variables
	variables
regularization_losses
	keras_api
(__call__
*)&call_and_return_all_conditional_losses"å
_tf_keras_layerË{"class_name": "FlexibleDense", "name": "flexible_dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 3]}}
<
0
	1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
	1
2
3"
trackable_list_wrapper
Ê
	variables
regularization_losses
non_trainable_variables
layer_metrics
layer_regularization_losses
metrics

layers
trainable_variables
#__call__
%_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
,
*serving_default"
signature_map
8:62&my_sequential_model/flexible_dense_1/w
4:22&my_sequential_model/flexible_dense_1/b
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
­

trainable_variables
	variables
layer_metrics
non_trainable_variables
layer_regularization_losses
metrics

layers
regularization_losses
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
8:62&my_sequential_model/flexible_dense_2/w
4:22&my_sequential_model/flexible_dense_2/b
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
trainable_variables
	variables
layer_metrics
non_trainable_variables
 layer_regularization_losses
!metrics

"layers
regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
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
ú2÷
1__inference_my_sequential_model_layer_call_fn_868Á
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
2
L__inference_my_sequential_model_layer_call_and_return_conditional_losses_854Á
²
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
Ü2Ù
__inference__wrapped_model_797¶
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
Ø2Õ
.__inference_flexible_dense_1_layer_call_fn_902¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_flexible_dense_1_layer_call_and_return_conditional_losses_893¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Ø2Õ
.__inference_flexible_dense_2_layer_call_fn_921¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ó2ð
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_912¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÈBÅ
!__inference_signature_wrapper_883input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
__inference__wrapped_model_797m	0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ©
I__inference_flexible_dense_1_layer_call_and_return_conditional_losses_893\	/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_flexible_dense_1_layer_call_fn_902O	/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ©
I__inference_flexible_dense_2_layer_call_and_return_conditional_losses_912\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_flexible_dense_2_layer_call_fn_921O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¯
L__inference_my_sequential_model_layer_call_and_return_conditional_losses_854_	0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
1__inference_my_sequential_model_layer_call_fn_868R	0¢-
&¢#
!
input_1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
!__inference_signature_wrapper_883x	;¢8
¢ 
1ª.
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ"3ª0
.
output_1"
output_1ÿÿÿÿÿÿÿÿÿ