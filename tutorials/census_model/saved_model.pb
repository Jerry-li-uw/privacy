??&
??
8
Const
output"dtype"
valuetensor"
dtypetype
l
LookupTableExportV2
table_handle
keys"Tkeys
values"Tvalues"
Tkeystype"
Tvaluestype?
?
MutableHashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?

NoOp
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
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
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.3.12v2.3.0-54-gfcc4b966f18??
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
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
?
sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:|@*(
shared_namesequential/dense/kernel
?
+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel*
_output_shapes

:|@*
dtype0
?
sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_namesequential/dense/bias
{
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes
:@*
dtype0
?
sequential/predictions/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*.
shared_namesequential/predictions/kernel
?
1sequential/predictions/kernel/Read/ReadVariableOpReadVariableOpsequential/predictions/kernel*
_output_shapes

:@*
dtype0
?
sequential/predictions/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namesequential/predictions/bias
?
/sequential/predictions/bias/Read/ReadVariableOpReadVariableOpsequential/predictions/bias*
_output_shapes
:*
dtype0
?
string_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_54*
value_dtype0	
?
string_lookup_1_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_86*
value_dtype0	
?
string_lookup_2_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_118*
value_dtype0	
?
string_lookup_3_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_150*
value_dtype0	
?
string_lookup_4_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_182*
value_dtype0	
?
string_lookup_5_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_214*
value_dtype0	
?
string_lookup_6_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_246*
value_dtype0	
?
string_lookup_7_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_278*
value_dtype0	
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
Adam/sequential/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:|@*/
shared_name Adam/sequential/dense/kernel/m
?
2Adam/sequential/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/kernel/m*
_output_shapes

:|@*
dtype0
?
Adam/sequential/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/sequential/dense/bias/m
?
0Adam/sequential/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/bias/m*
_output_shapes
:@*
dtype0
?
$Adam/sequential/predictions/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*5
shared_name&$Adam/sequential/predictions/kernel/m
?
8Adam/sequential/predictions/kernel/m/Read/ReadVariableOpReadVariableOp$Adam/sequential/predictions/kernel/m*
_output_shapes

:@*
dtype0
?
"Adam/sequential/predictions/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/sequential/predictions/bias/m
?
6Adam/sequential/predictions/bias/m/Read/ReadVariableOpReadVariableOp"Adam/sequential/predictions/bias/m*
_output_shapes
:*
dtype0
?
Adam/sequential/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:|@*/
shared_name Adam/sequential/dense/kernel/v
?
2Adam/sequential/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/kernel/v*
_output_shapes

:|@*
dtype0
?
Adam/sequential/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_nameAdam/sequential/dense/bias/v
?
0Adam/sequential/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/sequential/dense/bias/v*
_output_shapes
:@*
dtype0
?
$Adam/sequential/predictions/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*5
shared_name&$Adam/sequential/predictions/kernel/v
?
8Adam/sequential/predictions/kernel/v/Read/ReadVariableOpReadVariableOp$Adam/sequential/predictions/kernel/v*
_output_shapes

:@*
dtype0
?
"Adam/sequential/predictions/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/sequential/predictions/bias/v
?
6Adam/sequential/predictions/bias/v/Read/ReadVariableOpReadVariableOp"Adam/sequential/predictions/bias/v*
_output_shapes
:*
dtype0
G
ConstConst*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_1Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_2Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_3Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_4Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_5Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_6Const*
_output_shapes
: *
dtype0	*
value	B	 R
I
Const_7Const*
_output_shapes
: *
dtype0	*
value	B	 R
?
PartitionedCallPartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_15373
?
PartitionedCall_1PartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_15378
?
PartitionedCall_2PartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_15383
?
PartitionedCall_3PartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_15388
?
PartitionedCall_4PartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_15393
?
PartitionedCall_5PartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_15398
?
PartitionedCall_6PartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_15403
?
PartitionedCall_7PartitionedCall*	
Tin
 *
Tout
2*
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
GPU 2J 8? *#
fR
__inference_<lambda>_15408
?
NoOpNoOp^PartitionedCall^PartitionedCall_1^PartitionedCall_2^PartitionedCall_3^PartitionedCall_4^PartitionedCall_5^PartitionedCall_6^PartitionedCall_7
?
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_index_table*
Tkeys0*
Tvalues0	*,
_class"
 loc:@string_lookup_index_table*2
_output_shapes 
:?????????:?????????
?
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_1_index_table*
Tkeys0*
Tvalues0	*.
_class$
" loc:@string_lookup_1_index_table*2
_output_shapes 
:?????????:?????????
?
Jstring_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_2_index_table*
Tkeys0*
Tvalues0	*.
_class$
" loc:@string_lookup_2_index_table*2
_output_shapes 
:?????????:?????????
?
Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_3_index_table*
Tkeys0*
Tvalues0	*.
_class$
" loc:@string_lookup_3_index_table*2
_output_shapes 
:?????????:?????????
?
Jstring_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_4_index_table*
Tkeys0*
Tvalues0	*.
_class$
" loc:@string_lookup_4_index_table*2
_output_shapes 
:?????????:?????????
?
Jstring_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_5_index_table*
Tkeys0*
Tvalues0	*.
_class$
" loc:@string_lookup_5_index_table*2
_output_shapes 
:?????????:?????????
?
Jstring_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_6_index_table*
Tkeys0*
Tvalues0	*.
_class$
" loc:@string_lookup_6_index_table*2
_output_shapes 
:?????????:?????????
?
Jstring_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2string_lookup_7_index_table*
Tkeys0*
Tvalues0	*.
_class$
" loc:@string_lookup_7_index_table*2
_output_shapes 
:?????????:?????????
?<
Const_8Const"/device:CPU:0*
_output_shapes
: *
dtype0*?<
value?<B?< B?<
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-0
layer-14
layer_with_weights-1
layer-15
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
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
?
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
layer-8

layer-9
layer-10
layer-11
layer-12
	layer-13
layer-14
layer_with_weights-0
layer-15
layer_with_weights-1
layer-16
layer_with_weights-2
layer-17
layer_with_weights-3
layer-18
layer_with_weights-4
layer-19
layer_with_weights-5
layer-20
layer_with_weights-6
layer-21
layer_with_weights-7
layer-22
 layer_with_weights-8
 layer-23
!layer-24
"layer-25
#layer-26
$layer-27
%layer-28
&layer-29
'layer-30
(layer-31
)layer-32
*	variables
+regularization_losses
,trainable_variables
-	keras_api
?
.layer_with_weights-0
.layer-0
/layer_with_weights-1
/layer-1
0	variables
1regularization_losses
2trainable_variables
3	keras_api
?
4iter

5beta_1

6beta_2
	7decay
8learning_rate<m?=m?>m??m?<v?=v?>v??v?
6
98
:9
;10
<11
=12
>13
?14
 

<0
=1
>2
?3
?

@layers
	variables
Ametrics
Blayer_metrics
regularization_losses
Clayer_regularization_losses
Dnon_trainable_variables
trainable_variables
 
R
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
0
Istate_variables

J_table
K	keras_api
0
Lstate_variables

M_table
N	keras_api
0
Ostate_variables

P_table
Q	keras_api
0
Rstate_variables

S_table
T	keras_api
0
Ustate_variables

V_table
W	keras_api
0
Xstate_variables

Y_table
Z	keras_api
0
[state_variables

\_table
]	keras_api
0
^state_variables

__table
`	keras_api
]
astate_variables
b_broadcast_shape
9mean
:variance
	;count
c	keras_api
$
dstate_variables
e	keras_api
$
fstate_variables
g	keras_api
$
hstate_variables
i	keras_api
$
jstate_variables
k	keras_api
$
lstate_variables
m	keras_api
$
nstate_variables
o	keras_api
$
pstate_variables
q	keras_api
$
rstate_variables
s	keras_api
R
t	variables
uregularization_losses
vtrainable_variables
w	keras_api

98
:9
;10
 
 
?

xlayers
*	variables
ymetrics
zlayer_metrics
+regularization_losses
{layer_regularization_losses
|non_trainable_variables
,trainable_variables
~
}_inbound_nodes

<kernel
=bias
~	variables
regularization_losses
?trainable_variables
?	keras_api
?
?_inbound_nodes

>kernel
?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api

<0
=1
>2
?3
 

<0
=1
>2
?3
?
?layers
0	variables
?metrics
?layer_metrics
1regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
2trainable_variables
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
@>
VARIABLE_VALUEmean&variables/8/.ATTRIBUTES/VARIABLE_VALUE
DB
VARIABLE_VALUEvariance&variables/9/.ATTRIBUTES/VARIABLE_VALUE
B@
VARIABLE_VALUEcount'variables/10/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEsequential/dense/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEsequential/dense/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEsequential/predictions/kernel'variables/13/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEsequential/predictions/bias'variables/14/.ATTRIBUTES/VARIABLE_VALUE
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15

?0
 
 

98
:9
;10
 
 
 
?
?layers
E	variables
?metrics
?layer_metrics
Fregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
Gtrainable_variables
 
MK
tableBlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table
 
 
MK
tableBlayer_with_weights-0/layer_with_weights-1/_table/.ATTRIBUTES/table
 
 
MK
tableBlayer_with_weights-0/layer_with_weights-2/_table/.ATTRIBUTES/table
 
 
MK
tableBlayer_with_weights-0/layer_with_weights-3/_table/.ATTRIBUTES/table
 
 
MK
tableBlayer_with_weights-0/layer_with_weights-4/_table/.ATTRIBUTES/table
 
 
MK
tableBlayer_with_weights-0/layer_with_weights-5/_table/.ATTRIBUTES/table
 
 
MK
tableBlayer_with_weights-0/layer_with_weights-6/_table/.ATTRIBUTES/table
 
 
MK
tableBlayer_with_weights-0/layer_with_weights-7/_table/.ATTRIBUTES/table
 
#
9mean
:variance
	;count
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
?
?layers
t	variables
?metrics
?layer_metrics
uregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
vtrainable_variables
?
0
1
2
3
4
5
6
7
8

9
10
11
12
	13
14
15
16
17
18
19
20
21
22
 23
!24
"25
#26
$27
%28
&29
'30
(31
)32
 
 
 

98
:9
;10
 

<0
=1
 

<0
=1
?
?layers
~	variables
?metrics
?layer_metrics
regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
 

>0
?1
 

>0
?1
?
?layers
?	variables
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables

.0
/1
 
 
 
 
8

?total

?count
?	variables
?	keras_api
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
wu
VARIABLE_VALUEAdam/sequential/dense/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/sequential/dense/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$Adam/sequential/predictions/kernel/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential/predictions/bias/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/sequential/dense/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/sequential/dense/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE$Adam/sequential/predictions/kernel/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE"Adam/sequential/predictions/bias/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
v
serving_default_agePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_capital-gainPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_capital-lossPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_educationPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_education-numPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
y
serving_default_fnlwgtPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_hours-per-weekPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_marital-statusPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_native-countryPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
}
serving_default_occupationPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
w
serving_default_racePlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????

serving_default_relationshipPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
v
serving_default_sexPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
|
serving_default_workclassPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_ageserving_default_capital-gainserving_default_capital-lossserving_default_educationserving_default_education-numserving_default_fnlwgtserving_default_hours-per-weekserving_default_marital-statusserving_default_native-countryserving_default_occupationserving_default_raceserving_default_relationshipserving_default_sexserving_default_workclassstring_lookup_7_index_tableConststring_lookup_6_index_tableConst_1string_lookup_5_index_tableConst_2string_lookup_4_index_tableConst_3string_lookup_3_index_tableConst_4string_lookup_2_index_tableConst_5string_lookup_1_index_tableConst_6string_lookup_index_tableConst_7meanvariancesequential/dense/kernelsequential/dense/biassequential/predictions/kernelsequential/predictions/bias*/
Tin(
&2$								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_13707
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpmean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOp1sequential/predictions/kernel/Read/ReadVariableOp/sequential/predictions/bias/Read/ReadVariableOpHstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:1Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:1Jstring_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2:1Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2:1Jstring_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2:1Jstring_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2:1Jstring_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2:1Jstring_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2Lstring_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2:1total/Read/ReadVariableOpcount_1/Read/ReadVariableOp2Adam/sequential/dense/kernel/m/Read/ReadVariableOp0Adam/sequential/dense/bias/m/Read/ReadVariableOp8Adam/sequential/predictions/kernel/m/Read/ReadVariableOp6Adam/sequential/predictions/bias/m/Read/ReadVariableOp2Adam/sequential/dense/kernel/v/Read/ReadVariableOp0Adam/sequential/dense/bias/v/Read/ReadVariableOp8Adam/sequential/predictions/kernel/v/Read/ReadVariableOp6Adam/sequential/predictions/bias/v/Read/ReadVariableOpConst_8*3
Tin,
*2(										*
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
GPU 2J 8? *'
f"R 
__inference__traced_save_15566
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratemeanvariancecountsequential/dense/kernelsequential/dense/biassequential/predictions/kernelsequential/predictions/biasstring_lookup_index_tablestring_lookup_1_index_tablestring_lookup_2_index_tablestring_lookup_3_index_tablestring_lookup_4_index_tablestring_lookup_5_index_tablestring_lookup_6_index_tablestring_lookup_7_index_tabletotalcount_1Adam/sequential/dense/kernel/mAdam/sequential/dense/bias/m$Adam/sequential/predictions/kernel/m"Adam/sequential/predictions/bias/mAdam/sequential/dense/kernel/vAdam/sequential/dense/bias/v$Adam/sequential/predictions/kernel/v"Adam/sequential/predictions/bias/v**
Tin#
!2*
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
GPU 2J 8? **
f%R#
!__inference__traced_restore_15666??
?
K
__inference__creator_15097
identity??string_lookup_4_index_table?
string_lookup_4_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_182*
value_dtype0	2
string_lookup_4_index_table?
IdentityIdentity*string_lookup_4_index_table:table_handle:0^string_lookup_4_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 2:
string_lookup_4_index_tablestring_lookup_4_index_table
?
?
F__inference_predictions_layer_call_and_return_conditional_losses_13037

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
+__inference_predictions_layer_call_fn_15032

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_130372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
F__inference_predictions_layer_call_and_return_conditional_losses_15023

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmaxe
IdentityIdentitySoftmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????@:::O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?
G__inference_functional_1_layer_call_and_return_conditional_losses_12539
age
capital_gain
capital_loss
	education
education_num

fnlwgt
hours_per_week
marital_status
native_country

occupation
race
relationship
sex
	workclassI
Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	G
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
identity??6string_lookup/None_lookup_table_find/LookupTableFindV2?8string_lookup_1/None_lookup_table_find/LookupTableFindV2?8string_lookup_2/None_lookup_table_find/LookupTableFindV2?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?8string_lookup_4/None_lookup_table_find/LookupTableFindV2?8string_lookup_5/None_lookup_table_find/LookupTableFindV2?8string_lookup_6/None_lookup_table_find/LookupTableFindV2?8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_7/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handlenative_countryFstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_6/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handlesexFstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_6/None_lookup_table_find/LookupTableFindV2?
8string_lookup_5/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleraceFstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_5/None_lookup_table_find/LookupTableFindV2?
8string_lookup_4/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handlerelationshipFstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_4/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handle
occupationFstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_2/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handlemarital_statusFstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_2/None_lookup_table_find/LookupTableFindV2?
8string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle	educationFstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handle	workclassDstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2?
concatenate/PartitionedCallPartitionedCallagefnlwgteducation_numcapital_gaincapital_losshours_per_week*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_121902
concatenate/PartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSub$concatenate/PartitionedCall:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrt?
normalization/truedivRealDivnormalization/sub:z:0normalization/Sqrt:y:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Maximum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
"category_encoding_1/bincount/ShapeShapeAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_1/bincount/Shape?
"category_encoding_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_1/bincount/Const?
!category_encoding_1/bincount/ProdProd+category_encoding_1/bincount/Shape:output:0+category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_1/bincount/Prod?
&category_encoding_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_1/bincount/Greater/y?
$category_encoding_1/bincount/GreaterGreater*category_encoding_1/bincount/Prod:output:0/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_1/bincount/Greater?
!category_encoding_1/bincount/CastCast(category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_1/bincount/Cast?
$category_encoding_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_1/bincount/Const_1?
 category_encoding_1/bincount/MaxMaxAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/Max?
"category_encoding_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_1/bincount/add/y?
 category_encoding_1/bincount/addAddV2)category_encoding_1/bincount/Max:output:0+category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/add?
 category_encoding_1/bincount/mulMul%category_encoding_1/bincount/Cast:y:0$category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/mul?
&category_encoding_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_1/bincount/minlength?
$category_encoding_1/bincount/MaximumMaximum/category_encoding_1/bincount/minlength:output:0$category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_1/bincount/Maximum?
$category_encoding_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_1/bincount/Const_2?
*category_encoding_1/bincount/DenseBincountDenseBincountAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_1/bincount/Maximum:z:0-category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_1/bincount/DenseBincount?
"category_encoding_2/bincount/ShapeShapeAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_2/bincount/Shape?
"category_encoding_2/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_2/bincount/Const?
!category_encoding_2/bincount/ProdProd+category_encoding_2/bincount/Shape:output:0+category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_2/bincount/Prod?
&category_encoding_2/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_2/bincount/Greater/y?
$category_encoding_2/bincount/GreaterGreater*category_encoding_2/bincount/Prod:output:0/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_2/bincount/Greater?
!category_encoding_2/bincount/CastCast(category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_2/bincount/Cast?
$category_encoding_2/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_2/bincount/Const_1?
 category_encoding_2/bincount/MaxMaxAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/Max?
"category_encoding_2/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_2/bincount/add/y?
 category_encoding_2/bincount/addAddV2)category_encoding_2/bincount/Max:output:0+category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/add?
 category_encoding_2/bincount/mulMul%category_encoding_2/bincount/Cast:y:0$category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/mul?
&category_encoding_2/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R	2(
&category_encoding_2/bincount/minlength?
$category_encoding_2/bincount/MaximumMaximum/category_encoding_2/bincount/minlength:output:0$category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_2/bincount/Maximum?
$category_encoding_2/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_2/bincount/Const_2?
*category_encoding_2/bincount/DenseBincountDenseBincountAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_2/bincount/Maximum:z:0-category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????	*
binary_output(2,
*category_encoding_2/bincount/DenseBincount?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Maximum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
"category_encoding_4/bincount/ShapeShapeAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_4/bincount/Shape?
"category_encoding_4/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_4/bincount/Const?
!category_encoding_4/bincount/ProdProd+category_encoding_4/bincount/Shape:output:0+category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_4/bincount/Prod?
&category_encoding_4/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_4/bincount/Greater/y?
$category_encoding_4/bincount/GreaterGreater*category_encoding_4/bincount/Prod:output:0/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_4/bincount/Greater?
!category_encoding_4/bincount/CastCast(category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_4/bincount/Cast?
$category_encoding_4/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_4/bincount/Const_1?
 category_encoding_4/bincount/MaxMaxAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/Max?
"category_encoding_4/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_4/bincount/add/y?
 category_encoding_4/bincount/addAddV2)category_encoding_4/bincount/Max:output:0+category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/add?
 category_encoding_4/bincount/mulMul%category_encoding_4/bincount/Cast:y:0$category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/mul?
&category_encoding_4/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_4/bincount/minlength?
$category_encoding_4/bincount/MaximumMaximum/category_encoding_4/bincount/minlength:output:0$category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_4/bincount/Maximum?
$category_encoding_4/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_4/bincount/Const_2?
*category_encoding_4/bincount/DenseBincountDenseBincountAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_4/bincount/Maximum:z:0-category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_4/bincount/DenseBincount?
"category_encoding_5/bincount/ShapeShapeAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_5/bincount/Shape?
"category_encoding_5/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_5/bincount/Const?
!category_encoding_5/bincount/ProdProd+category_encoding_5/bincount/Shape:output:0+category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_5/bincount/Prod?
&category_encoding_5/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_5/bincount/Greater/y?
$category_encoding_5/bincount/GreaterGreater*category_encoding_5/bincount/Prod:output:0/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_5/bincount/Greater?
!category_encoding_5/bincount/CastCast(category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_5/bincount/Cast?
$category_encoding_5/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_5/bincount/Const_1?
 category_encoding_5/bincount/MaxMaxAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/Max?
"category_encoding_5/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_5/bincount/add/y?
 category_encoding_5/bincount/addAddV2)category_encoding_5/bincount/Max:output:0+category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/add?
 category_encoding_5/bincount/mulMul%category_encoding_5/bincount/Cast:y:0$category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/mul?
&category_encoding_5/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_5/bincount/minlength?
$category_encoding_5/bincount/MaximumMaximum/category_encoding_5/bincount/minlength:output:0$category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_5/bincount/Maximum?
$category_encoding_5/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_5/bincount/Const_2?
*category_encoding_5/bincount/DenseBincountDenseBincountAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_5/bincount/Maximum:z:0-category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_5/bincount/DenseBincount?
"category_encoding_6/bincount/ShapeShapeAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_6/bincount/Shape?
"category_encoding_6/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_6/bincount/Const?
!category_encoding_6/bincount/ProdProd+category_encoding_6/bincount/Shape:output:0+category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_6/bincount/Prod?
&category_encoding_6/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_6/bincount/Greater/y?
$category_encoding_6/bincount/GreaterGreater*category_encoding_6/bincount/Prod:output:0/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_6/bincount/Greater?
!category_encoding_6/bincount/CastCast(category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_6/bincount/Cast?
$category_encoding_6/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_6/bincount/Const_1?
 category_encoding_6/bincount/MaxMaxAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/Max?
"category_encoding_6/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_6/bincount/add/y?
 category_encoding_6/bincount/addAddV2)category_encoding_6/bincount/Max:output:0+category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/add?
 category_encoding_6/bincount/mulMul%category_encoding_6/bincount/Cast:y:0$category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/mul?
&category_encoding_6/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_6/bincount/minlength?
$category_encoding_6/bincount/MaximumMaximum/category_encoding_6/bincount/minlength:output:0$category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_6/bincount/Maximum?
$category_encoding_6/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_6/bincount/Const_2?
*category_encoding_6/bincount/DenseBincountDenseBincountAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_6/bincount/Maximum:z:0-category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_6/bincount/DenseBincount?
"category_encoding_7/bincount/ShapeShapeAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_7/bincount/Shape?
"category_encoding_7/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_7/bincount/Const?
!category_encoding_7/bincount/ProdProd+category_encoding_7/bincount/Shape:output:0+category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_7/bincount/Prod?
&category_encoding_7/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_7/bincount/Greater/y?
$category_encoding_7/bincount/GreaterGreater*category_encoding_7/bincount/Prod:output:0/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_7/bincount/Greater?
!category_encoding_7/bincount/CastCast(category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_7/bincount/Cast?
$category_encoding_7/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_7/bincount/Const_1?
 category_encoding_7/bincount/MaxMaxAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/Max?
"category_encoding_7/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_7/bincount/add/y?
 category_encoding_7/bincount/addAddV2)category_encoding_7/bincount/Max:output:0+category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/add?
 category_encoding_7/bincount/mulMul%category_encoding_7/bincount/Cast:y:0$category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/mul?
&category_encoding_7/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2(
&category_encoding_7/bincount/minlength?
$category_encoding_7/bincount/MaximumMaximum/category_encoding_7/bincount/minlength:output:0$category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_7/bincount/Maximum?
$category_encoding_7/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_7/bincount/Const_2?
*category_encoding_7/bincount/DenseBincountDenseBincountAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_7/bincount/Maximum:z:0-category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2,
*category_encoding_7/bincount/DenseBincount?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:01category_encoding/bincount/DenseBincount:output:03category_encoding_1/bincount/DenseBincount:output:03category_encoding_2/bincount/DenseBincount:output:03category_encoding_3/bincount/DenseBincount:output:03category_encoding_4/bincount/DenseBincount:output:03category_encoding_5/bincount/DenseBincount:output:03category_encoding_6/bincount/DenseBincount:output:03category_encoding_7/bincount/DenseBincount:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_123482
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:07^string_lookup/None_lookup_table_find/LookupTableFindV29^string_lookup_1/None_lookup_table_find/LookupTableFindV29^string_lookup_2/None_lookup_table_find/LookupTableFindV29^string_lookup_3/None_lookup_table_find/LookupTableFindV29^string_lookup_4/None_lookup_table_find/LookupTableFindV29^string_lookup_5/None_lookup_table_find/LookupTableFindV29^string_lookup_6/None_lookup_table_find/LookupTableFindV29^string_lookup_7/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV22t
8string_lookup_1/None_lookup_table_find/LookupTableFindV28string_lookup_1/None_lookup_table_find/LookupTableFindV22t
8string_lookup_2/None_lookup_table_find/LookupTableFindV28string_lookup_2/None_lookup_table_find/LookupTableFindV22t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV22t
8string_lookup_4/None_lookup_table_find/LookupTableFindV28string_lookup_4/None_lookup_table_find/LookupTableFindV22t
8string_lookup_5/None_lookup_table_find/LookupTableFindV28string_lookup_5/None_lookup_table_find/LookupTableFindV22t
8string_lookup_6/None_lookup_table_find/LookupTableFindV28string_lookup_6/None_lookup_table_find/LookupTableFindV22t
8string_lookup_7/None_lookup_table_find/LookupTableFindV28string_lookup_7/None_lookup_table_find/LookupTableFindV2:L H
'
_output_shapes
:?????????

_user_specified_nameage:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-gain:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-loss:RN
'
_output_shapes
:?????????
#
_user_specified_name	education:VR
'
_output_shapes
:?????????
'
_user_specified_nameeducation-num:OK
'
_output_shapes
:?????????
 
_user_specified_namefnlwgt:WS
'
_output_shapes
:?????????
(
_user_specified_namehours-per-week:WS
'
_output_shapes
:?????????
(
_user_specified_namemarital-status:WS
'
_output_shapes
:?????????
(
_user_specified_namenative-country:S	O
'
_output_shapes
:?????????
$
_user_specified_name
occupation:M
I
'
_output_shapes
:?????????

_user_specified_namerace:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelationship:LH
'
_output_shapes
:?????????

_user_specified_namesex:RN
'
_output_shapes
:?????????
#
_user_specified_name	workclass
?
.
__inference__initializer_15087
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?!
?
G__inference_functional_3_layer_call_and_return_conditional_losses_13321
age
capital_gain
capital_loss
	education
education_num

fnlwgt
hours_per_week
marital_status
native_country

occupation
race
relationship
sex
	workclass
functional_1_13248
functional_1_13250	
functional_1_13252
functional_1_13254	
functional_1_13256
functional_1_13258	
functional_1_13260
functional_1_13262	
functional_1_13264
functional_1_13266	
functional_1_13268
functional_1_13270	
functional_1_13272
functional_1_13274	
functional_1_13276
functional_1_13278	
functional_1_13280
functional_1_13282
sequential_13311
sequential_13313
sequential_13315
sequential_13317
identity??$functional_1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
$functional_1/StatefulPartitionedCallStatefulPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclassfunctional_1_13248functional_1_13250functional_1_13252functional_1_13254functional_1_13256functional_1_13258functional_1_13260functional_1_13262functional_1_13264functional_1_13266functional_1_13268functional_1_13270functional_1_13272functional_1_13274functional_1_13276functional_1_13278functional_1_13280functional_1_13282*+
Tin$
"2 								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_127292&
$functional_1/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall-functional_1/StatefulPartitionedCall:output:0sequential_13311sequential_13313sequential_13315sequential_13317*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_130852$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0%^functional_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::2L
$functional_1/StatefulPartitionedCall$functional_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
'
_output_shapes
:?????????

_user_specified_nameage:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-gain:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-loss:RN
'
_output_shapes
:?????????
#
_user_specified_name	education:VR
'
_output_shapes
:?????????
'
_user_specified_nameeducation-num:OK
'
_output_shapes
:?????????
 
_user_specified_namefnlwgt:WS
'
_output_shapes
:?????????
(
_user_specified_namehours-per-week:WS
'
_output_shapes
:?????????
(
_user_specified_namemarital-status:WS
'
_output_shapes
:?????????
(
_user_specified_namenative-country:S	O
'
_output_shapes
:?????????
$
_user_specified_name
occupation:M
I
'
_output_shapes
:?????????

_user_specified_namerace:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelationship:LH
'
_output_shapes
:?????????

_user_specified_namesex:RN
'
_output_shapes
:?????????
#
_user_specified_name	workclass
?
,
__inference__destroyer_15062
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
,
__inference__destroyer_15077
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
*
__inference_<lambda>_15398
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
__inference_save_fn_15171
checkpoint_keyY
Ustring_lookup_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2?
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Ustring_lookup_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*'
_output_shapes
:?????????:2J
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityOstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:keys:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*#
_output_shapes
:?????????2

Identity_2?

Identity_3Identity	add_1:z:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentityQstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:values:0I^string_lookup_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2Hstring_lookup_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_14902

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource.
*predictions_matmul_readvariableop_resource/
+predictions_biasadd_readvariableop_resource
identity??
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:|@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAdd?
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!predictions/MatMul/ReadVariableOp?
predictions/MatMulMatMuldense/BiasAdd:output:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
predictions/MatMul?
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"predictions/BiasAdd/ReadVariableOp?
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
predictions/BiasAdd?
predictions/SoftmaxSoftmaxpredictions/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
predictions/Softmaxq
IdentityIdentitypredictions/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????|:::::O K
'
_output_shapes
:?????????|
 
_user_specified_nameinputs
?
.
__inference__initializer_15072
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_13085

inputs
dense_13074
dense_13076
predictions_13079
predictions_13081
identity??dense/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_13074dense_13076*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_130102
dense/StatefulPartitionedCall?
#predictions/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0predictions_13079predictions_13081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_130372%
#predictions/StatefulPartitionedCall?
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????|::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall:O K
'
_output_shapes
:?????????|
 
_user_specified_nameinputs
?
*
__inference_<lambda>_15408
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?	
?
__inference_restore_fn_15314
restored_tensors_0
restored_tensors_1	N
Jstring_lookup_5_index_table_table_restore_lookuptableimportv2_table_handle
identity??=string_lookup_5_index_table_table_restore/LookupTableImportV2?
=string_lookup_5_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_5_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2?
=string_lookup_5_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0>^string_lookup_5_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::2~
=string_lookup_5_index_table_table_restore/LookupTableImportV2=string_lookup_5_index_table_table_restore/LookupTableImportV2:W S
#
_output_shapes
:?????????
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?	
?
__inference_restore_fn_15368
restored_tensors_0
restored_tensors_1	N
Jstring_lookup_7_index_table_table_restore_lookuptableimportv2_table_handle
identity??=string_lookup_7_index_table_table_restore/LookupTableImportV2?
=string_lookup_7_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_7_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2?
=string_lookup_7_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0>^string_lookup_7_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::2~
=string_lookup_7_index_table_table_restore/LookupTableImportV2=string_lookup_7_index_table_table_restore/LookupTableImportV2:W S
#
_output_shapes
:?????????
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
?
#__inference_signature_wrapper_13707
age
capital_gain
capital_loss
	education
education_num

fnlwgt
hours_per_week
marital_status
native_country

occupation
race
relationship
sex
	workclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*/
Tin(
&2$								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_121382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:?????????

_user_specified_nameage:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-gain:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-loss:RN
'
_output_shapes
:?????????
#
_user_specified_name	education:VR
'
_output_shapes
:?????????
'
_user_specified_nameeducation-num:OK
'
_output_shapes
:?????????
 
_user_specified_namefnlwgt:WS
'
_output_shapes
:?????????
(
_user_specified_namehours-per-week:WS
'
_output_shapes
:?????????
(
_user_specified_namemarital-status:WS
'
_output_shapes
:?????????
(
_user_specified_namenative-country:S	O
'
_output_shapes
:?????????
$
_user_specified_name
occupation:M
I
'
_output_shapes
:?????????

_user_specified_namerace:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelationship:LH
'
_output_shapes
:?????????

_user_specified_namesex:RN
'
_output_shapes
:?????????
#
_user_specified_name	workclass
?
*
__inference_<lambda>_15383
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
,__inference_functional_1_layer_call_fn_14885

inputs_age
inputs_capital_gain
inputs_capital_loss
inputs_education
inputs_education_num
inputs_fnlwgt
inputs_hours_per_week
inputs_marital_status
inputs_native_country
inputs_occupation
inputs_race
inputs_relationship

inputs_sex
inputs_workclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_ageinputs_capital_gaininputs_capital_lossinputs_educationinputs_education_numinputs_fnlwgtinputs_hours_per_weekinputs_marital_statusinputs_native_countryinputs_occupationinputs_raceinputs_relationship
inputs_sexinputs_workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*+
Tin$
"2 								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_129572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/age:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-gain:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-loss:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/education:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/education-num:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/fnlwgt:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/hours-per-week:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/marital-status:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/native-country:Z	V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:T
P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/race:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/relationship:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/workclass
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_13068
dense_input
dense_13057
dense_13059
predictions_13062
predictions_13064
identity??dense/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_13057dense_13059*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_130102
dense/StatefulPartitionedCall?
#predictions/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0predictions_13062predictions_13064*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_130372%
#predictions/StatefulPartitionedCall?
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????|::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall:T P
'
_output_shapes
:?????????|
%
_user_specified_namedense_input
?
?
@__inference_dense_layer_call_and_return_conditional_losses_13010

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:|@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????|:::O K
'
_output_shapes
:?????????|
 
_user_specified_nameinputs
??
?
G__inference_functional_3_layer_call_and_return_conditional_losses_14301

inputs_age
inputs_capital_gain
inputs_capital_loss
inputs_education
inputs_education_num
inputs_fnlwgt
inputs_hours_per_week
inputs_marital_status
inputs_native_country
inputs_occupation
inputs_race
inputs_relationship

inputs_sex
inputs_workclassV
Rfunctional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	T
Pfunctional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleU
Qfunctional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	>
:functional_1_normalization_reshape_readvariableop_resource@
<functional_1_normalization_reshape_1_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource9
5sequential_predictions_matmul_readvariableop_resource:
6sequential_predictions_biasadd_readvariableop_resource
identity??Cfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleinputs_native_countrySfunctional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handle
inputs_sexSfunctional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleinputs_raceSfunctional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleinputs_relationshipSfunctional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputs_occupationSfunctional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleinputs_marital_statusSfunctional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleinputs_educationSfunctional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
Cfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Pfunctional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleinputs_workclassQfunctional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2E
Cfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2?
$functional_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_1/concatenate/concat/axis?
functional_1/concatenate/concatConcatV2
inputs_ageinputs_fnlwgtinputs_education_numinputs_capital_gaininputs_capital_lossinputs_hours_per_week-functional_1/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2!
functional_1/concatenate/concat?
1functional_1/normalization/Reshape/ReadVariableOpReadVariableOp:functional_1_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype023
1functional_1/normalization/Reshape/ReadVariableOp?
(functional_1/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2*
(functional_1/normalization/Reshape/shape?
"functional_1/normalization/ReshapeReshape9functional_1/normalization/Reshape/ReadVariableOp:value:01functional_1/normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2$
"functional_1/normalization/Reshape?
3functional_1/normalization/Reshape_1/ReadVariableOpReadVariableOp<functional_1_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype025
3functional_1/normalization/Reshape_1/ReadVariableOp?
*functional_1/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*functional_1/normalization/Reshape_1/shape?
$functional_1/normalization/Reshape_1Reshape;functional_1/normalization/Reshape_1/ReadVariableOp:value:03functional_1/normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2&
$functional_1/normalization/Reshape_1?
functional_1/normalization/subSub(functional_1/concatenate/concat:output:0+functional_1/normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2 
functional_1/normalization/sub?
functional_1/normalization/SqrtSqrt-functional_1/normalization/Reshape_1:output:0*
T0*
_output_shapes

:2!
functional_1/normalization/Sqrt?
"functional_1/normalization/truedivRealDiv"functional_1/normalization/sub:z:0#functional_1/normalization/Sqrt:y:0*
T0*'
_output_shapes
:?????????2$
"functional_1/normalization/truediv?
-functional_1/category_encoding/bincount/ShapeShapeLfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2/
-functional_1/category_encoding/bincount/Shape?
-functional_1/category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/category_encoding/bincount/Const?
,functional_1/category_encoding/bincount/ProdProd6functional_1/category_encoding/bincount/Shape:output:06functional_1/category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2.
,functional_1/category_encoding/bincount/Prod?
1functional_1/category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 23
1functional_1/category_encoding/bincount/Greater/y?
/functional_1/category_encoding/bincount/GreaterGreater5functional_1/category_encoding/bincount/Prod:output:0:functional_1/category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 21
/functional_1/category_encoding/bincount/Greater?
,functional_1/category_encoding/bincount/CastCast3functional_1/category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2.
,functional_1/category_encoding/bincount/Cast?
/functional_1/category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/functional_1/category_encoding/bincount/Const_1?
+functional_1/category_encoding/bincount/MaxMaxLfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2:values:08functional_1/category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2-
+functional_1/category_encoding/bincount/Max?
-functional_1/category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2/
-functional_1/category_encoding/bincount/add/y?
+functional_1/category_encoding/bincount/addAddV24functional_1/category_encoding/bincount/Max:output:06functional_1/category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2-
+functional_1/category_encoding/bincount/add?
+functional_1/category_encoding/bincount/mulMul0functional_1/category_encoding/bincount/Cast:y:0/functional_1/category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2-
+functional_1/category_encoding/bincount/mul?
1functional_1/category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R23
1functional_1/category_encoding/bincount/minlength?
/functional_1/category_encoding/bincount/MaximumMaximum:functional_1/category_encoding/bincount/minlength:output:0/functional_1/category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 21
/functional_1/category_encoding/bincount/Maximum?
/functional_1/category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 21
/functional_1/category_encoding/bincount/Const_2?
5functional_1/category_encoding/bincount/DenseBincountDenseBincountLfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2:values:03functional_1/category_encoding/bincount/Maximum:z:08functional_1/category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(27
5functional_1/category_encoding/bincount/DenseBincount?
/functional_1/category_encoding_1/bincount/ShapeShapeNfunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_1/bincount/Shape?
/functional_1/category_encoding_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_1/bincount/Const?
.functional_1/category_encoding_1/bincount/ProdProd8functional_1/category_encoding_1/bincount/Shape:output:08functional_1/category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_1/bincount/Prod?
3functional_1/category_encoding_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_1/bincount/Greater/y?
1functional_1/category_encoding_1/bincount/GreaterGreater7functional_1/category_encoding_1/bincount/Prod:output:0<functional_1/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_1/bincount/Greater?
.functional_1/category_encoding_1/bincount/CastCast5functional_1/category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_1/bincount/Cast?
1functional_1/category_encoding_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_1/bincount/Const_1?
-functional_1/category_encoding_1/bincount/MaxMaxNfunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_1/bincount/Max?
/functional_1/category_encoding_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_1/bincount/add/y?
-functional_1/category_encoding_1/bincount/addAddV26functional_1/category_encoding_1/bincount/Max:output:08functional_1/category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_1/bincount/add?
-functional_1/category_encoding_1/bincount/mulMul2functional_1/category_encoding_1/bincount/Cast:y:01functional_1/category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_1/bincount/mul?
3functional_1/category_encoding_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3functional_1/category_encoding_1/bincount/minlength?
1functional_1/category_encoding_1/bincount/MaximumMaximum<functional_1/category_encoding_1/bincount/minlength:output:01functional_1/category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_1/bincount/Maximum?
1functional_1/category_encoding_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_1/bincount/Const_2?
7functional_1/category_encoding_1/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_1/bincount/Maximum:z:0:functional_1/category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(29
7functional_1/category_encoding_1/bincount/DenseBincount?
/functional_1/category_encoding_2/bincount/ShapeShapeNfunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_2/bincount/Shape?
/functional_1/category_encoding_2/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_2/bincount/Const?
.functional_1/category_encoding_2/bincount/ProdProd8functional_1/category_encoding_2/bincount/Shape:output:08functional_1/category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_2/bincount/Prod?
3functional_1/category_encoding_2/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_2/bincount/Greater/y?
1functional_1/category_encoding_2/bincount/GreaterGreater7functional_1/category_encoding_2/bincount/Prod:output:0<functional_1/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_2/bincount/Greater?
.functional_1/category_encoding_2/bincount/CastCast5functional_1/category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_2/bincount/Cast?
1functional_1/category_encoding_2/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_2/bincount/Const_1?
-functional_1/category_encoding_2/bincount/MaxMaxNfunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_2/bincount/Max?
/functional_1/category_encoding_2/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_2/bincount/add/y?
-functional_1/category_encoding_2/bincount/addAddV26functional_1/category_encoding_2/bincount/Max:output:08functional_1/category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_2/bincount/add?
-functional_1/category_encoding_2/bincount/mulMul2functional_1/category_encoding_2/bincount/Cast:y:01functional_1/category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_2/bincount/mul?
3functional_1/category_encoding_2/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R	25
3functional_1/category_encoding_2/bincount/minlength?
1functional_1/category_encoding_2/bincount/MaximumMaximum<functional_1/category_encoding_2/bincount/minlength:output:01functional_1/category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_2/bincount/Maximum?
1functional_1/category_encoding_2/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_2/bincount/Const_2?
7functional_1/category_encoding_2/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_2/bincount/Maximum:z:0:functional_1/category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????	*
binary_output(29
7functional_1/category_encoding_2/bincount/DenseBincount?
/functional_1/category_encoding_3/bincount/ShapeShapeNfunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_3/bincount/Shape?
/functional_1/category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_3/bincount/Const?
.functional_1/category_encoding_3/bincount/ProdProd8functional_1/category_encoding_3/bincount/Shape:output:08functional_1/category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_3/bincount/Prod?
3functional_1/category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_3/bincount/Greater/y?
1functional_1/category_encoding_3/bincount/GreaterGreater7functional_1/category_encoding_3/bincount/Prod:output:0<functional_1/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_3/bincount/Greater?
.functional_1/category_encoding_3/bincount/CastCast5functional_1/category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_3/bincount/Cast?
1functional_1/category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_3/bincount/Const_1?
-functional_1/category_encoding_3/bincount/MaxMaxNfunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_3/bincount/Max?
/functional_1/category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_3/bincount/add/y?
-functional_1/category_encoding_3/bincount/addAddV26functional_1/category_encoding_3/bincount/Max:output:08functional_1/category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_3/bincount/add?
-functional_1/category_encoding_3/bincount/mulMul2functional_1/category_encoding_3/bincount/Cast:y:01functional_1/category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_3/bincount/mul?
3functional_1/category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3functional_1/category_encoding_3/bincount/minlength?
1functional_1/category_encoding_3/bincount/MaximumMaximum<functional_1/category_encoding_3/bincount/minlength:output:01functional_1/category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_3/bincount/Maximum?
1functional_1/category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_3/bincount/Const_2?
7functional_1/category_encoding_3/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_3/bincount/Maximum:z:0:functional_1/category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(29
7functional_1/category_encoding_3/bincount/DenseBincount?
/functional_1/category_encoding_4/bincount/ShapeShapeNfunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_4/bincount/Shape?
/functional_1/category_encoding_4/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_4/bincount/Const?
.functional_1/category_encoding_4/bincount/ProdProd8functional_1/category_encoding_4/bincount/Shape:output:08functional_1/category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_4/bincount/Prod?
3functional_1/category_encoding_4/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_4/bincount/Greater/y?
1functional_1/category_encoding_4/bincount/GreaterGreater7functional_1/category_encoding_4/bincount/Prod:output:0<functional_1/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_4/bincount/Greater?
.functional_1/category_encoding_4/bincount/CastCast5functional_1/category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_4/bincount/Cast?
1functional_1/category_encoding_4/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_4/bincount/Const_1?
-functional_1/category_encoding_4/bincount/MaxMaxNfunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_4/bincount/Max?
/functional_1/category_encoding_4/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_4/bincount/add/y?
-functional_1/category_encoding_4/bincount/addAddV26functional_1/category_encoding_4/bincount/Max:output:08functional_1/category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_4/bincount/add?
-functional_1/category_encoding_4/bincount/mulMul2functional_1/category_encoding_4/bincount/Cast:y:01functional_1/category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_4/bincount/mul?
3functional_1/category_encoding_4/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3functional_1/category_encoding_4/bincount/minlength?
1functional_1/category_encoding_4/bincount/MaximumMaximum<functional_1/category_encoding_4/bincount/minlength:output:01functional_1/category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_4/bincount/Maximum?
1functional_1/category_encoding_4/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_4/bincount/Const_2?
7functional_1/category_encoding_4/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_4/bincount/Maximum:z:0:functional_1/category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(29
7functional_1/category_encoding_4/bincount/DenseBincount?
/functional_1/category_encoding_5/bincount/ShapeShapeNfunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_5/bincount/Shape?
/functional_1/category_encoding_5/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_5/bincount/Const?
.functional_1/category_encoding_5/bincount/ProdProd8functional_1/category_encoding_5/bincount/Shape:output:08functional_1/category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_5/bincount/Prod?
3functional_1/category_encoding_5/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_5/bincount/Greater/y?
1functional_1/category_encoding_5/bincount/GreaterGreater7functional_1/category_encoding_5/bincount/Prod:output:0<functional_1/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_5/bincount/Greater?
.functional_1/category_encoding_5/bincount/CastCast5functional_1/category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_5/bincount/Cast?
1functional_1/category_encoding_5/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_5/bincount/Const_1?
-functional_1/category_encoding_5/bincount/MaxMaxNfunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_5/bincount/Max?
/functional_1/category_encoding_5/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_5/bincount/add/y?
-functional_1/category_encoding_5/bincount/addAddV26functional_1/category_encoding_5/bincount/Max:output:08functional_1/category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_5/bincount/add?
-functional_1/category_encoding_5/bincount/mulMul2functional_1/category_encoding_5/bincount/Cast:y:01functional_1/category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_5/bincount/mul?
3functional_1/category_encoding_5/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3functional_1/category_encoding_5/bincount/minlength?
1functional_1/category_encoding_5/bincount/MaximumMaximum<functional_1/category_encoding_5/bincount/minlength:output:01functional_1/category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_5/bincount/Maximum?
1functional_1/category_encoding_5/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_5/bincount/Const_2?
7functional_1/category_encoding_5/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_5/bincount/Maximum:z:0:functional_1/category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(29
7functional_1/category_encoding_5/bincount/DenseBincount?
/functional_1/category_encoding_6/bincount/ShapeShapeNfunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_6/bincount/Shape?
/functional_1/category_encoding_6/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_6/bincount/Const?
.functional_1/category_encoding_6/bincount/ProdProd8functional_1/category_encoding_6/bincount/Shape:output:08functional_1/category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_6/bincount/Prod?
3functional_1/category_encoding_6/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_6/bincount/Greater/y?
1functional_1/category_encoding_6/bincount/GreaterGreater7functional_1/category_encoding_6/bincount/Prod:output:0<functional_1/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_6/bincount/Greater?
.functional_1/category_encoding_6/bincount/CastCast5functional_1/category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_6/bincount/Cast?
1functional_1/category_encoding_6/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_6/bincount/Const_1?
-functional_1/category_encoding_6/bincount/MaxMaxNfunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_6/bincount/Max?
/functional_1/category_encoding_6/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_6/bincount/add/y?
-functional_1/category_encoding_6/bincount/addAddV26functional_1/category_encoding_6/bincount/Max:output:08functional_1/category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_6/bincount/add?
-functional_1/category_encoding_6/bincount/mulMul2functional_1/category_encoding_6/bincount/Cast:y:01functional_1/category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_6/bincount/mul?
3functional_1/category_encoding_6/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3functional_1/category_encoding_6/bincount/minlength?
1functional_1/category_encoding_6/bincount/MaximumMaximum<functional_1/category_encoding_6/bincount/minlength:output:01functional_1/category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_6/bincount/Maximum?
1functional_1/category_encoding_6/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_6/bincount/Const_2?
7functional_1/category_encoding_6/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_6/bincount/Maximum:z:0:functional_1/category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(29
7functional_1/category_encoding_6/bincount/DenseBincount?
/functional_1/category_encoding_7/bincount/ShapeShapeNfunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_7/bincount/Shape?
/functional_1/category_encoding_7/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_7/bincount/Const?
.functional_1/category_encoding_7/bincount/ProdProd8functional_1/category_encoding_7/bincount/Shape:output:08functional_1/category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_7/bincount/Prod?
3functional_1/category_encoding_7/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_7/bincount/Greater/y?
1functional_1/category_encoding_7/bincount/GreaterGreater7functional_1/category_encoding_7/bincount/Prod:output:0<functional_1/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_7/bincount/Greater?
.functional_1/category_encoding_7/bincount/CastCast5functional_1/category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_7/bincount/Cast?
1functional_1/category_encoding_7/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_7/bincount/Const_1?
-functional_1/category_encoding_7/bincount/MaxMaxNfunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_7/bincount/Max?
/functional_1/category_encoding_7/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_7/bincount/add/y?
-functional_1/category_encoding_7/bincount/addAddV26functional_1/category_encoding_7/bincount/Max:output:08functional_1/category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_7/bincount/add?
-functional_1/category_encoding_7/bincount/mulMul2functional_1/category_encoding_7/bincount/Cast:y:01functional_1/category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_7/bincount/mul?
3functional_1/category_encoding_7/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,25
3functional_1/category_encoding_7/bincount/minlength?
1functional_1/category_encoding_7/bincount/MaximumMaximum<functional_1/category_encoding_7/bincount/minlength:output:01functional_1/category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_7/bincount/Maximum?
1functional_1/category_encoding_7/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_7/bincount/Const_2?
7functional_1/category_encoding_7/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_7/bincount/Maximum:z:0:functional_1/category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(29
7functional_1/category_encoding_7/bincount/DenseBincount?
&functional_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&functional_1/concatenate_1/concat/axis?
!functional_1/concatenate_1/concatConcatV2&functional_1/normalization/truediv:z:0>functional_1/category_encoding/bincount/DenseBincount:output:0@functional_1/category_encoding_1/bincount/DenseBincount:output:0@functional_1/category_encoding_2/bincount/DenseBincount:output:0@functional_1/category_encoding_3/bincount/DenseBincount:output:0@functional_1/category_encoding_4/bincount/DenseBincount:output:0@functional_1/category_encoding_5/bincount/DenseBincount:output:0@functional_1/category_encoding_6/bincount/DenseBincount:output:0@functional_1/category_encoding_7/bincount/DenseBincount:output:0/functional_1/concatenate_1/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????|2#
!functional_1/concatenate_1/concat?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:|@*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul*functional_1/concatenate_1/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense/BiasAdd?
,sequential/predictions/MatMul/ReadVariableOpReadVariableOp5sequential_predictions_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,sequential/predictions/MatMul/ReadVariableOp?
sequential/predictions/MatMulMatMul!sequential/dense/BiasAdd:output:04sequential/predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/predictions/MatMul?
-sequential/predictions/BiasAdd/ReadVariableOpReadVariableOp6sequential_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/predictions/BiasAdd/ReadVariableOp?
sequential/predictions/BiasAddBiasAdd'sequential/predictions/MatMul:product:05sequential/predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential/predictions/BiasAdd?
sequential/predictions/SoftmaxSoftmax'sequential/predictions/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential/predictions/Softmax?
IdentityIdentity(sequential/predictions/Softmax:softmax:0D^functional_1/string_lookup/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::2?
Cfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2Cfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/age:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-gain:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-loss:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/education:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/education-num:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/fnlwgt:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/hours-per-week:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/marital-status:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/native-country:Z	V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:T
P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/race:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/relationship:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/workclass
?
.
__inference__initializer_15132
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
*__inference_sequential_layer_call_fn_13096
dense_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_130852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????|::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????|
%
_user_specified_namedense_input
?
*
__inference_<lambda>_15393
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
*__inference_sequential_layer_call_fn_14945

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_131122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????|::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????|
 
_user_specified_nameinputs
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_13054
dense_input
dense_13021
dense_13023
predictions_13048
predictions_13050
identity??dense/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCalldense_inputdense_13021dense_13023*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_130102
dense/StatefulPartitionedCall?
#predictions/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0predictions_13048predictions_13050*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_130372%
#predictions/StatefulPartitionedCall?
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????|::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall:T P
'
_output_shapes
:?????????|
%
_user_specified_namedense_input
?
?
__inference_save_fn_15333
checkpoint_key[
Wstring_lookup_6_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Jstring_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2?
Jstring_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_6_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*'
_output_shapes
:?????????:2L
Jstring_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0K^string_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0K^string_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityQstring_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2*
T0*#
_output_shapes
:?????????2

Identity_2?

Identity_3Identity	add_1:z:0K^string_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0K^string_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentitySstring_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Jstring_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_6_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
,__inference_functional_1_layer_call_fn_12768
age
capital_gain
capital_loss
	education
education_num

fnlwgt
hours_per_week
marital_status
native_country

occupation
race
relationship
sex
	workclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*+
Tin$
"2 								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_127292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:?????????

_user_specified_nameage:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-gain:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-loss:RN
'
_output_shapes
:?????????
#
_user_specified_name	education:VR
'
_output_shapes
:?????????
'
_user_specified_nameeducation-num:OK
'
_output_shapes
:?????????
 
_user_specified_namefnlwgt:WS
'
_output_shapes
:?????????
(
_user_specified_namehours-per-week:WS
'
_output_shapes
:?????????
(
_user_specified_namemarital-status:WS
'
_output_shapes
:?????????
(
_user_specified_namenative-country:S	O
'
_output_shapes
:?????????
$
_user_specified_name
occupation:M
I
'
_output_shapes
:?????????

_user_specified_namerace:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelationship:LH
'
_output_shapes
:?????????

_user_specified_namesex:RN
'
_output_shapes
:?????????
#
_user_specified_name	workclass
?	
?
F__inference_concatenate_layer_call_and_return_conditional_losses_14956
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5
??
?
G__inference_functional_3_layer_call_and_return_conditional_losses_14112

inputs_age
inputs_capital_gain
inputs_capital_loss
inputs_education
inputs_education_num
inputs_fnlwgt
inputs_hours_per_week
inputs_marital_status
inputs_native_country
inputs_occupation
inputs_race
inputs_relationship

inputs_sex
inputs_workclassV
Rfunctional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value	V
Rfunctional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleW
Sfunctional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	T
Pfunctional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleU
Qfunctional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	>
:functional_1_normalization_reshape_readvariableop_resource@
<functional_1_normalization_reshape_1_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource9
5sequential_predictions_matmul_readvariableop_resource:
6sequential_predictions_biasadd_readvariableop_resource
identity??Cfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2?Efunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleinputs_native_countrySfunctional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handle
inputs_sexSfunctional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleinputs_raceSfunctional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleinputs_relationshipSfunctional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputs_occupationSfunctional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleinputs_marital_statusSfunctional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2?
Efunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Rfunctional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleinputs_educationSfunctional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2G
Efunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
Cfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Pfunctional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleinputs_workclassQfunctional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2E
Cfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2?
$functional_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2&
$functional_1/concatenate/concat/axis?
functional_1/concatenate/concatConcatV2
inputs_ageinputs_fnlwgtinputs_education_numinputs_capital_gaininputs_capital_lossinputs_hours_per_week-functional_1/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2!
functional_1/concatenate/concat?
1functional_1/normalization/Reshape/ReadVariableOpReadVariableOp:functional_1_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype023
1functional_1/normalization/Reshape/ReadVariableOp?
(functional_1/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2*
(functional_1/normalization/Reshape/shape?
"functional_1/normalization/ReshapeReshape9functional_1/normalization/Reshape/ReadVariableOp:value:01functional_1/normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2$
"functional_1/normalization/Reshape?
3functional_1/normalization/Reshape_1/ReadVariableOpReadVariableOp<functional_1_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype025
3functional_1/normalization/Reshape_1/ReadVariableOp?
*functional_1/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2,
*functional_1/normalization/Reshape_1/shape?
$functional_1/normalization/Reshape_1Reshape;functional_1/normalization/Reshape_1/ReadVariableOp:value:03functional_1/normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2&
$functional_1/normalization/Reshape_1?
functional_1/normalization/subSub(functional_1/concatenate/concat:output:0+functional_1/normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2 
functional_1/normalization/sub?
functional_1/normalization/SqrtSqrt-functional_1/normalization/Reshape_1:output:0*
T0*
_output_shapes

:2!
functional_1/normalization/Sqrt?
"functional_1/normalization/truedivRealDiv"functional_1/normalization/sub:z:0#functional_1/normalization/Sqrt:y:0*
T0*'
_output_shapes
:?????????2$
"functional_1/normalization/truediv?
-functional_1/category_encoding/bincount/ShapeShapeLfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2/
-functional_1/category_encoding/bincount/Shape?
-functional_1/category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2/
-functional_1/category_encoding/bincount/Const?
,functional_1/category_encoding/bincount/ProdProd6functional_1/category_encoding/bincount/Shape:output:06functional_1/category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2.
,functional_1/category_encoding/bincount/Prod?
1functional_1/category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 23
1functional_1/category_encoding/bincount/Greater/y?
/functional_1/category_encoding/bincount/GreaterGreater5functional_1/category_encoding/bincount/Prod:output:0:functional_1/category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 21
/functional_1/category_encoding/bincount/Greater?
,functional_1/category_encoding/bincount/CastCast3functional_1/category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2.
,functional_1/category_encoding/bincount/Cast?
/functional_1/category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       21
/functional_1/category_encoding/bincount/Const_1?
+functional_1/category_encoding/bincount/MaxMaxLfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2:values:08functional_1/category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2-
+functional_1/category_encoding/bincount/Max?
-functional_1/category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2/
-functional_1/category_encoding/bincount/add/y?
+functional_1/category_encoding/bincount/addAddV24functional_1/category_encoding/bincount/Max:output:06functional_1/category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2-
+functional_1/category_encoding/bincount/add?
+functional_1/category_encoding/bincount/mulMul0functional_1/category_encoding/bincount/Cast:y:0/functional_1/category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2-
+functional_1/category_encoding/bincount/mul?
1functional_1/category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R23
1functional_1/category_encoding/bincount/minlength?
/functional_1/category_encoding/bincount/MaximumMaximum:functional_1/category_encoding/bincount/minlength:output:0/functional_1/category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 21
/functional_1/category_encoding/bincount/Maximum?
/functional_1/category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 21
/functional_1/category_encoding/bincount/Const_2?
5functional_1/category_encoding/bincount/DenseBincountDenseBincountLfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2:values:03functional_1/category_encoding/bincount/Maximum:z:08functional_1/category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(27
5functional_1/category_encoding/bincount/DenseBincount?
/functional_1/category_encoding_1/bincount/ShapeShapeNfunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_1/bincount/Shape?
/functional_1/category_encoding_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_1/bincount/Const?
.functional_1/category_encoding_1/bincount/ProdProd8functional_1/category_encoding_1/bincount/Shape:output:08functional_1/category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_1/bincount/Prod?
3functional_1/category_encoding_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_1/bincount/Greater/y?
1functional_1/category_encoding_1/bincount/GreaterGreater7functional_1/category_encoding_1/bincount/Prod:output:0<functional_1/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_1/bincount/Greater?
.functional_1/category_encoding_1/bincount/CastCast5functional_1/category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_1/bincount/Cast?
1functional_1/category_encoding_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_1/bincount/Const_1?
-functional_1/category_encoding_1/bincount/MaxMaxNfunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_1/bincount/Max?
/functional_1/category_encoding_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_1/bincount/add/y?
-functional_1/category_encoding_1/bincount/addAddV26functional_1/category_encoding_1/bincount/Max:output:08functional_1/category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_1/bincount/add?
-functional_1/category_encoding_1/bincount/mulMul2functional_1/category_encoding_1/bincount/Cast:y:01functional_1/category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_1/bincount/mul?
3functional_1/category_encoding_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3functional_1/category_encoding_1/bincount/minlength?
1functional_1/category_encoding_1/bincount/MaximumMaximum<functional_1/category_encoding_1/bincount/minlength:output:01functional_1/category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_1/bincount/Maximum?
1functional_1/category_encoding_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_1/bincount/Const_2?
7functional_1/category_encoding_1/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_1/bincount/Maximum:z:0:functional_1/category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(29
7functional_1/category_encoding_1/bincount/DenseBincount?
/functional_1/category_encoding_2/bincount/ShapeShapeNfunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_2/bincount/Shape?
/functional_1/category_encoding_2/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_2/bincount/Const?
.functional_1/category_encoding_2/bincount/ProdProd8functional_1/category_encoding_2/bincount/Shape:output:08functional_1/category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_2/bincount/Prod?
3functional_1/category_encoding_2/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_2/bincount/Greater/y?
1functional_1/category_encoding_2/bincount/GreaterGreater7functional_1/category_encoding_2/bincount/Prod:output:0<functional_1/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_2/bincount/Greater?
.functional_1/category_encoding_2/bincount/CastCast5functional_1/category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_2/bincount/Cast?
1functional_1/category_encoding_2/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_2/bincount/Const_1?
-functional_1/category_encoding_2/bincount/MaxMaxNfunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_2/bincount/Max?
/functional_1/category_encoding_2/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_2/bincount/add/y?
-functional_1/category_encoding_2/bincount/addAddV26functional_1/category_encoding_2/bincount/Max:output:08functional_1/category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_2/bincount/add?
-functional_1/category_encoding_2/bincount/mulMul2functional_1/category_encoding_2/bincount/Cast:y:01functional_1/category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_2/bincount/mul?
3functional_1/category_encoding_2/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R	25
3functional_1/category_encoding_2/bincount/minlength?
1functional_1/category_encoding_2/bincount/MaximumMaximum<functional_1/category_encoding_2/bincount/minlength:output:01functional_1/category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_2/bincount/Maximum?
1functional_1/category_encoding_2/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_2/bincount/Const_2?
7functional_1/category_encoding_2/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_2/bincount/Maximum:z:0:functional_1/category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????	*
binary_output(29
7functional_1/category_encoding_2/bincount/DenseBincount?
/functional_1/category_encoding_3/bincount/ShapeShapeNfunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_3/bincount/Shape?
/functional_1/category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_3/bincount/Const?
.functional_1/category_encoding_3/bincount/ProdProd8functional_1/category_encoding_3/bincount/Shape:output:08functional_1/category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_3/bincount/Prod?
3functional_1/category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_3/bincount/Greater/y?
1functional_1/category_encoding_3/bincount/GreaterGreater7functional_1/category_encoding_3/bincount/Prod:output:0<functional_1/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_3/bincount/Greater?
.functional_1/category_encoding_3/bincount/CastCast5functional_1/category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_3/bincount/Cast?
1functional_1/category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_3/bincount/Const_1?
-functional_1/category_encoding_3/bincount/MaxMaxNfunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_3/bincount/Max?
/functional_1/category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_3/bincount/add/y?
-functional_1/category_encoding_3/bincount/addAddV26functional_1/category_encoding_3/bincount/Max:output:08functional_1/category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_3/bincount/add?
-functional_1/category_encoding_3/bincount/mulMul2functional_1/category_encoding_3/bincount/Cast:y:01functional_1/category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_3/bincount/mul?
3functional_1/category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3functional_1/category_encoding_3/bincount/minlength?
1functional_1/category_encoding_3/bincount/MaximumMaximum<functional_1/category_encoding_3/bincount/minlength:output:01functional_1/category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_3/bincount/Maximum?
1functional_1/category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_3/bincount/Const_2?
7functional_1/category_encoding_3/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_3/bincount/Maximum:z:0:functional_1/category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(29
7functional_1/category_encoding_3/bincount/DenseBincount?
/functional_1/category_encoding_4/bincount/ShapeShapeNfunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_4/bincount/Shape?
/functional_1/category_encoding_4/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_4/bincount/Const?
.functional_1/category_encoding_4/bincount/ProdProd8functional_1/category_encoding_4/bincount/Shape:output:08functional_1/category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_4/bincount/Prod?
3functional_1/category_encoding_4/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_4/bincount/Greater/y?
1functional_1/category_encoding_4/bincount/GreaterGreater7functional_1/category_encoding_4/bincount/Prod:output:0<functional_1/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_4/bincount/Greater?
.functional_1/category_encoding_4/bincount/CastCast5functional_1/category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_4/bincount/Cast?
1functional_1/category_encoding_4/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_4/bincount/Const_1?
-functional_1/category_encoding_4/bincount/MaxMaxNfunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_4/bincount/Max?
/functional_1/category_encoding_4/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_4/bincount/add/y?
-functional_1/category_encoding_4/bincount/addAddV26functional_1/category_encoding_4/bincount/Max:output:08functional_1/category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_4/bincount/add?
-functional_1/category_encoding_4/bincount/mulMul2functional_1/category_encoding_4/bincount/Cast:y:01functional_1/category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_4/bincount/mul?
3functional_1/category_encoding_4/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3functional_1/category_encoding_4/bincount/minlength?
1functional_1/category_encoding_4/bincount/MaximumMaximum<functional_1/category_encoding_4/bincount/minlength:output:01functional_1/category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_4/bincount/Maximum?
1functional_1/category_encoding_4/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_4/bincount/Const_2?
7functional_1/category_encoding_4/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_4/bincount/Maximum:z:0:functional_1/category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(29
7functional_1/category_encoding_4/bincount/DenseBincount?
/functional_1/category_encoding_5/bincount/ShapeShapeNfunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_5/bincount/Shape?
/functional_1/category_encoding_5/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_5/bincount/Const?
.functional_1/category_encoding_5/bincount/ProdProd8functional_1/category_encoding_5/bincount/Shape:output:08functional_1/category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_5/bincount/Prod?
3functional_1/category_encoding_5/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_5/bincount/Greater/y?
1functional_1/category_encoding_5/bincount/GreaterGreater7functional_1/category_encoding_5/bincount/Prod:output:0<functional_1/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_5/bincount/Greater?
.functional_1/category_encoding_5/bincount/CastCast5functional_1/category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_5/bincount/Cast?
1functional_1/category_encoding_5/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_5/bincount/Const_1?
-functional_1/category_encoding_5/bincount/MaxMaxNfunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_5/bincount/Max?
/functional_1/category_encoding_5/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_5/bincount/add/y?
-functional_1/category_encoding_5/bincount/addAddV26functional_1/category_encoding_5/bincount/Max:output:08functional_1/category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_5/bincount/add?
-functional_1/category_encoding_5/bincount/mulMul2functional_1/category_encoding_5/bincount/Cast:y:01functional_1/category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_5/bincount/mul?
3functional_1/category_encoding_5/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3functional_1/category_encoding_5/bincount/minlength?
1functional_1/category_encoding_5/bincount/MaximumMaximum<functional_1/category_encoding_5/bincount/minlength:output:01functional_1/category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_5/bincount/Maximum?
1functional_1/category_encoding_5/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_5/bincount/Const_2?
7functional_1/category_encoding_5/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_5/bincount/Maximum:z:0:functional_1/category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(29
7functional_1/category_encoding_5/bincount/DenseBincount?
/functional_1/category_encoding_6/bincount/ShapeShapeNfunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_6/bincount/Shape?
/functional_1/category_encoding_6/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_6/bincount/Const?
.functional_1/category_encoding_6/bincount/ProdProd8functional_1/category_encoding_6/bincount/Shape:output:08functional_1/category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_6/bincount/Prod?
3functional_1/category_encoding_6/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_6/bincount/Greater/y?
1functional_1/category_encoding_6/bincount/GreaterGreater7functional_1/category_encoding_6/bincount/Prod:output:0<functional_1/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_6/bincount/Greater?
.functional_1/category_encoding_6/bincount/CastCast5functional_1/category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_6/bincount/Cast?
1functional_1/category_encoding_6/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_6/bincount/Const_1?
-functional_1/category_encoding_6/bincount/MaxMaxNfunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_6/bincount/Max?
/functional_1/category_encoding_6/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_6/bincount/add/y?
-functional_1/category_encoding_6/bincount/addAddV26functional_1/category_encoding_6/bincount/Max:output:08functional_1/category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_6/bincount/add?
-functional_1/category_encoding_6/bincount/mulMul2functional_1/category_encoding_6/bincount/Cast:y:01functional_1/category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_6/bincount/mul?
3functional_1/category_encoding_6/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R25
3functional_1/category_encoding_6/bincount/minlength?
1functional_1/category_encoding_6/bincount/MaximumMaximum<functional_1/category_encoding_6/bincount/minlength:output:01functional_1/category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_6/bincount/Maximum?
1functional_1/category_encoding_6/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_6/bincount/Const_2?
7functional_1/category_encoding_6/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_6/bincount/Maximum:z:0:functional_1/category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(29
7functional_1/category_encoding_6/bincount/DenseBincount?
/functional_1/category_encoding_7/bincount/ShapeShapeNfunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:21
/functional_1/category_encoding_7/bincount/Shape?
/functional_1/category_encoding_7/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 21
/functional_1/category_encoding_7/bincount/Const?
.functional_1/category_encoding_7/bincount/ProdProd8functional_1/category_encoding_7/bincount/Shape:output:08functional_1/category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 20
.functional_1/category_encoding_7/bincount/Prod?
3functional_1/category_encoding_7/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 25
3functional_1/category_encoding_7/bincount/Greater/y?
1functional_1/category_encoding_7/bincount/GreaterGreater7functional_1/category_encoding_7/bincount/Prod:output:0<functional_1/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 23
1functional_1/category_encoding_7/bincount/Greater?
.functional_1/category_encoding_7/bincount/CastCast5functional_1/category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 20
.functional_1/category_encoding_7/bincount/Cast?
1functional_1/category_encoding_7/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       23
1functional_1/category_encoding_7/bincount/Const_1?
-functional_1/category_encoding_7/bincount/MaxMaxNfunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0:functional_1/category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_7/bincount/Max?
/functional_1/category_encoding_7/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R21
/functional_1/category_encoding_7/bincount/add/y?
-functional_1/category_encoding_7/bincount/addAddV26functional_1/category_encoding_7/bincount/Max:output:08functional_1/category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_7/bincount/add?
-functional_1/category_encoding_7/bincount/mulMul2functional_1/category_encoding_7/bincount/Cast:y:01functional_1/category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2/
-functional_1/category_encoding_7/bincount/mul?
3functional_1/category_encoding_7/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,25
3functional_1/category_encoding_7/bincount/minlength?
1functional_1/category_encoding_7/bincount/MaximumMaximum<functional_1/category_encoding_7/bincount/minlength:output:01functional_1/category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 23
1functional_1/category_encoding_7/bincount/Maximum?
1functional_1/category_encoding_7/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 23
1functional_1/category_encoding_7/bincount/Const_2?
7functional_1/category_encoding_7/bincount/DenseBincountDenseBincountNfunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:values:05functional_1/category_encoding_7/bincount/Maximum:z:0:functional_1/category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(29
7functional_1/category_encoding_7/bincount/DenseBincount?
&functional_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2(
&functional_1/concatenate_1/concat/axis?
!functional_1/concatenate_1/concatConcatV2&functional_1/normalization/truediv:z:0>functional_1/category_encoding/bincount/DenseBincount:output:0@functional_1/category_encoding_1/bincount/DenseBincount:output:0@functional_1/category_encoding_2/bincount/DenseBincount:output:0@functional_1/category_encoding_3/bincount/DenseBincount:output:0@functional_1/category_encoding_4/bincount/DenseBincount:output:0@functional_1/category_encoding_5/bincount/DenseBincount:output:0@functional_1/category_encoding_6/bincount/DenseBincount:output:0@functional_1/category_encoding_7/bincount/DenseBincount:output:0/functional_1/concatenate_1/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????|2#
!functional_1/concatenate_1/concat?
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource*
_output_shapes

:|@*
dtype02(
&sequential/dense/MatMul/ReadVariableOp?
sequential/dense/MatMulMatMul*functional_1/concatenate_1/concat:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense/MatMul?
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'sequential/dense/BiasAdd/ReadVariableOp?
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
sequential/dense/BiasAdd?
,sequential/predictions/MatMul/ReadVariableOpReadVariableOp5sequential_predictions_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02.
,sequential/predictions/MatMul/ReadVariableOp?
sequential/predictions/MatMulMatMul!sequential/dense/BiasAdd:output:04sequential/predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
sequential/predictions/MatMul?
-sequential/predictions/BiasAdd/ReadVariableOpReadVariableOp6sequential_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02/
-sequential/predictions/BiasAdd/ReadVariableOp?
sequential/predictions/BiasAddBiasAdd'sequential/predictions/MatMul:product:05sequential/predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2 
sequential/predictions/BiasAdd?
sequential/predictions/SoftmaxSoftmax'sequential/predictions/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2 
sequential/predictions/Softmax?
IdentityIdentity(sequential/predictions/Softmax:softmax:0D^functional_1/string_lookup/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2F^functional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::2?
Cfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV2Cfunctional_1/string_lookup/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV22?
Efunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2Efunctional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/age:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-gain:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-loss:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/education:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/education-num:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/fnlwgt:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/hours-per-week:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/marital-status:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/native-country:Z	V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:T
P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/race:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/relationship:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/workclass
?	
?
__inference_restore_fn_15260
restored_tensors_0
restored_tensors_1	N
Jstring_lookup_3_index_table_table_restore_lookuptableimportv2_table_handle
identity??=string_lookup_3_index_table_table_restore/LookupTableImportV2?
=string_lookup_3_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_3_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2?
=string_lookup_3_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0>^string_lookup_3_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::2~
=string_lookup_3_index_table_table_restore/LookupTableImportV2=string_lookup_3_index_table_table_restore/LookupTableImportV2:W S
#
_output_shapes
:?????????
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_14980
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????|2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????	:?????????:?????????:?????????:?????????:?????????,:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????	
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????,
"
_user_specified_name
inputs/8
?
?
*__inference_sequential_layer_call_fn_13123
dense_input
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCalldense_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_131122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????|::::22
StatefulPartitionedCallStatefulPartitionedCall:T P
'
_output_shapes
:?????????|
%
_user_specified_namedense_input
?
?
,__inference_functional_3_layer_call_fn_14425

inputs_age
inputs_capital_gain
inputs_capital_loss
inputs_education
inputs_education_num
inputs_fnlwgt
inputs_hours_per_week
inputs_marital_status
inputs_native_country
inputs_occupation
inputs_race
inputs_relationship

inputs_sex
inputs_workclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_ageinputs_capital_gaininputs_capital_lossinputs_educationinputs_education_numinputs_fnlwgtinputs_hours_per_weekinputs_marital_statusinputs_native_countryinputs_occupationinputs_raceinputs_relationship
inputs_sexinputs_workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*/
Tin(
&2$								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_3_layer_call_and_return_conditional_losses_135882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/age:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-gain:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-loss:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/education:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/education-num:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/fnlwgt:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/hours-per-week:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/marital-status:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/native-country:Z	V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:T
P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/race:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/relationship:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/workclass
??
?
G__inference_functional_1_layer_call_and_return_conditional_losses_12729

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13I
Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	G
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
identity??6string_lookup/None_lookup_table_find/LookupTableFindV2?8string_lookup_1/None_lookup_table_find/LookupTableFindV2?8string_lookup_2/None_lookup_table_find/LookupTableFindV2?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?8string_lookup_4/None_lookup_table_find/LookupTableFindV2?8string_lookup_5/None_lookup_table_find/LookupTableFindV2?8string_lookup_6/None_lookup_table_find/LookupTableFindV2?8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_7/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleinputs_8Fstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_6/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handle	inputs_12Fstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_6/None_lookup_table_find/LookupTableFindV2?
8string_lookup_5/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handle	inputs_10Fstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_5/None_lookup_table_find/LookupTableFindV2?
8string_lookup_4/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handle	inputs_11Fstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_4/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputs_9Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_2/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleinputs_7Fstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_2/None_lookup_table_find/LookupTableFindV2?
8string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleinputs_3Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handle	inputs_13Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2?
concatenate/PartitionedCallPartitionedCallinputsinputs_5inputs_4inputs_1inputs_2inputs_6*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_121902
concatenate/PartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSub$concatenate/PartitionedCall:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrt?
normalization/truedivRealDivnormalization/sub:z:0normalization/Sqrt:y:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Maximum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
"category_encoding_1/bincount/ShapeShapeAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_1/bincount/Shape?
"category_encoding_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_1/bincount/Const?
!category_encoding_1/bincount/ProdProd+category_encoding_1/bincount/Shape:output:0+category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_1/bincount/Prod?
&category_encoding_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_1/bincount/Greater/y?
$category_encoding_1/bincount/GreaterGreater*category_encoding_1/bincount/Prod:output:0/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_1/bincount/Greater?
!category_encoding_1/bincount/CastCast(category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_1/bincount/Cast?
$category_encoding_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_1/bincount/Const_1?
 category_encoding_1/bincount/MaxMaxAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/Max?
"category_encoding_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_1/bincount/add/y?
 category_encoding_1/bincount/addAddV2)category_encoding_1/bincount/Max:output:0+category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/add?
 category_encoding_1/bincount/mulMul%category_encoding_1/bincount/Cast:y:0$category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/mul?
&category_encoding_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_1/bincount/minlength?
$category_encoding_1/bincount/MaximumMaximum/category_encoding_1/bincount/minlength:output:0$category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_1/bincount/Maximum?
$category_encoding_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_1/bincount/Const_2?
*category_encoding_1/bincount/DenseBincountDenseBincountAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_1/bincount/Maximum:z:0-category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_1/bincount/DenseBincount?
"category_encoding_2/bincount/ShapeShapeAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_2/bincount/Shape?
"category_encoding_2/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_2/bincount/Const?
!category_encoding_2/bincount/ProdProd+category_encoding_2/bincount/Shape:output:0+category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_2/bincount/Prod?
&category_encoding_2/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_2/bincount/Greater/y?
$category_encoding_2/bincount/GreaterGreater*category_encoding_2/bincount/Prod:output:0/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_2/bincount/Greater?
!category_encoding_2/bincount/CastCast(category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_2/bincount/Cast?
$category_encoding_2/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_2/bincount/Const_1?
 category_encoding_2/bincount/MaxMaxAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/Max?
"category_encoding_2/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_2/bincount/add/y?
 category_encoding_2/bincount/addAddV2)category_encoding_2/bincount/Max:output:0+category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/add?
 category_encoding_2/bincount/mulMul%category_encoding_2/bincount/Cast:y:0$category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/mul?
&category_encoding_2/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R	2(
&category_encoding_2/bincount/minlength?
$category_encoding_2/bincount/MaximumMaximum/category_encoding_2/bincount/minlength:output:0$category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_2/bincount/Maximum?
$category_encoding_2/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_2/bincount/Const_2?
*category_encoding_2/bincount/DenseBincountDenseBincountAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_2/bincount/Maximum:z:0-category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????	*
binary_output(2,
*category_encoding_2/bincount/DenseBincount?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Maximum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
"category_encoding_4/bincount/ShapeShapeAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_4/bincount/Shape?
"category_encoding_4/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_4/bincount/Const?
!category_encoding_4/bincount/ProdProd+category_encoding_4/bincount/Shape:output:0+category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_4/bincount/Prod?
&category_encoding_4/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_4/bincount/Greater/y?
$category_encoding_4/bincount/GreaterGreater*category_encoding_4/bincount/Prod:output:0/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_4/bincount/Greater?
!category_encoding_4/bincount/CastCast(category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_4/bincount/Cast?
$category_encoding_4/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_4/bincount/Const_1?
 category_encoding_4/bincount/MaxMaxAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/Max?
"category_encoding_4/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_4/bincount/add/y?
 category_encoding_4/bincount/addAddV2)category_encoding_4/bincount/Max:output:0+category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/add?
 category_encoding_4/bincount/mulMul%category_encoding_4/bincount/Cast:y:0$category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/mul?
&category_encoding_4/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_4/bincount/minlength?
$category_encoding_4/bincount/MaximumMaximum/category_encoding_4/bincount/minlength:output:0$category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_4/bincount/Maximum?
$category_encoding_4/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_4/bincount/Const_2?
*category_encoding_4/bincount/DenseBincountDenseBincountAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_4/bincount/Maximum:z:0-category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_4/bincount/DenseBincount?
"category_encoding_5/bincount/ShapeShapeAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_5/bincount/Shape?
"category_encoding_5/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_5/bincount/Const?
!category_encoding_5/bincount/ProdProd+category_encoding_5/bincount/Shape:output:0+category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_5/bincount/Prod?
&category_encoding_5/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_5/bincount/Greater/y?
$category_encoding_5/bincount/GreaterGreater*category_encoding_5/bincount/Prod:output:0/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_5/bincount/Greater?
!category_encoding_5/bincount/CastCast(category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_5/bincount/Cast?
$category_encoding_5/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_5/bincount/Const_1?
 category_encoding_5/bincount/MaxMaxAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/Max?
"category_encoding_5/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_5/bincount/add/y?
 category_encoding_5/bincount/addAddV2)category_encoding_5/bincount/Max:output:0+category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/add?
 category_encoding_5/bincount/mulMul%category_encoding_5/bincount/Cast:y:0$category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/mul?
&category_encoding_5/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_5/bincount/minlength?
$category_encoding_5/bincount/MaximumMaximum/category_encoding_5/bincount/minlength:output:0$category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_5/bincount/Maximum?
$category_encoding_5/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_5/bincount/Const_2?
*category_encoding_5/bincount/DenseBincountDenseBincountAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_5/bincount/Maximum:z:0-category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_5/bincount/DenseBincount?
"category_encoding_6/bincount/ShapeShapeAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_6/bincount/Shape?
"category_encoding_6/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_6/bincount/Const?
!category_encoding_6/bincount/ProdProd+category_encoding_6/bincount/Shape:output:0+category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_6/bincount/Prod?
&category_encoding_6/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_6/bincount/Greater/y?
$category_encoding_6/bincount/GreaterGreater*category_encoding_6/bincount/Prod:output:0/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_6/bincount/Greater?
!category_encoding_6/bincount/CastCast(category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_6/bincount/Cast?
$category_encoding_6/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_6/bincount/Const_1?
 category_encoding_6/bincount/MaxMaxAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/Max?
"category_encoding_6/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_6/bincount/add/y?
 category_encoding_6/bincount/addAddV2)category_encoding_6/bincount/Max:output:0+category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/add?
 category_encoding_6/bincount/mulMul%category_encoding_6/bincount/Cast:y:0$category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/mul?
&category_encoding_6/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_6/bincount/minlength?
$category_encoding_6/bincount/MaximumMaximum/category_encoding_6/bincount/minlength:output:0$category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_6/bincount/Maximum?
$category_encoding_6/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_6/bincount/Const_2?
*category_encoding_6/bincount/DenseBincountDenseBincountAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_6/bincount/Maximum:z:0-category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_6/bincount/DenseBincount?
"category_encoding_7/bincount/ShapeShapeAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_7/bincount/Shape?
"category_encoding_7/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_7/bincount/Const?
!category_encoding_7/bincount/ProdProd+category_encoding_7/bincount/Shape:output:0+category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_7/bincount/Prod?
&category_encoding_7/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_7/bincount/Greater/y?
$category_encoding_7/bincount/GreaterGreater*category_encoding_7/bincount/Prod:output:0/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_7/bincount/Greater?
!category_encoding_7/bincount/CastCast(category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_7/bincount/Cast?
$category_encoding_7/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_7/bincount/Const_1?
 category_encoding_7/bincount/MaxMaxAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/Max?
"category_encoding_7/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_7/bincount/add/y?
 category_encoding_7/bincount/addAddV2)category_encoding_7/bincount/Max:output:0+category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/add?
 category_encoding_7/bincount/mulMul%category_encoding_7/bincount/Cast:y:0$category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/mul?
&category_encoding_7/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2(
&category_encoding_7/bincount/minlength?
$category_encoding_7/bincount/MaximumMaximum/category_encoding_7/bincount/minlength:output:0$category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_7/bincount/Maximum?
$category_encoding_7/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_7/bincount/Const_2?
*category_encoding_7/bincount/DenseBincountDenseBincountAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_7/bincount/Maximum:z:0-category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2,
*category_encoding_7/bincount/DenseBincount?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:01category_encoding/bincount/DenseBincount:output:03category_encoding_1/bincount/DenseBincount:output:03category_encoding_2/bincount/DenseBincount:output:03category_encoding_3/bincount/DenseBincount:output:03category_encoding_4/bincount/DenseBincount:output:03category_encoding_5/bincount/DenseBincount:output:03category_encoding_6/bincount/DenseBincount:output:03category_encoding_7/bincount/DenseBincount:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_123482
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:07^string_lookup/None_lookup_table_find/LookupTableFindV29^string_lookup_1/None_lookup_table_find/LookupTableFindV29^string_lookup_2/None_lookup_table_find/LookupTableFindV29^string_lookup_3/None_lookup_table_find/LookupTableFindV29^string_lookup_4/None_lookup_table_find/LookupTableFindV29^string_lookup_5/None_lookup_table_find/LookupTableFindV29^string_lookup_6/None_lookup_table_find/LookupTableFindV29^string_lookup_7/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV22t
8string_lookup_1/None_lookup_table_find/LookupTableFindV28string_lookup_1/None_lookup_table_find/LookupTableFindV22t
8string_lookup_2/None_lookup_table_find/LookupTableFindV28string_lookup_2/None_lookup_table_find/LookupTableFindV22t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV22t
8string_lookup_4/None_lookup_table_find/LookupTableFindV28string_lookup_4/None_lookup_table_find/LookupTableFindV22t
8string_lookup_5/None_lookup_table_find/LookupTableFindV28string_lookup_5/None_lookup_table_find/LookupTableFindV22t
8string_lookup_6/None_lookup_table_find/LookupTableFindV28string_lookup_6/None_lookup_table_find/LookupTableFindV22t
8string_lookup_7/None_lookup_table_find/LookupTableFindV28string_lookup_7/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_save_fn_15360
checkpoint_key[
Wstring_lookup_7_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Jstring_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2?
Jstring_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_7_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*'
_output_shapes
:?????????:2L
Jstring_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0K^string_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0K^string_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityQstring_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2*
T0*#
_output_shapes
:?????????2

Identity_2?

Identity_3Identity	add_1:z:0K^string_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0K^string_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentitySstring_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Jstring_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_7_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
__inference_save_fn_15279
checkpoint_key[
Wstring_lookup_4_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Jstring_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2?
Jstring_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_4_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*'
_output_shapes
:?????????:2L
Jstring_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0K^string_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0K^string_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityQstring_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2*
T0*#
_output_shapes
:?????????2

Identity_2?

Identity_3Identity	add_1:z:0K^string_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0K^string_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentitySstring_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Jstring_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_4_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
*
__inference_<lambda>_15373
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?	
?
__inference_restore_fn_15233
restored_tensors_0
restored_tensors_1	N
Jstring_lookup_2_index_table_table_restore_lookuptableimportv2_table_handle
identity??=string_lookup_2_index_table_table_restore/LookupTableImportV2?
=string_lookup_2_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_2_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2?
=string_lookup_2_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0>^string_lookup_2_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::2~
=string_lookup_2_index_table_table_restore/LookupTableImportV2=string_lookup_2_index_table_table_restore/LookupTableImportV2:W S
#
_output_shapes
:?????????
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
I
__inference__creator_15037
identity??string_lookup_index_table?
string_lookup_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_54*
value_dtype0	2
string_lookup_index_table?
IdentityIdentity(string_lookup_index_table:table_handle:0^string_lookup_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 26
string_lookup_index_tablestring_lookup_index_table
?	
?
__inference_restore_fn_15206
restored_tensors_0
restored_tensors_1	N
Jstring_lookup_1_index_table_table_restore_lookuptableimportv2_table_handle
identity??=string_lookup_1_index_table_table_restore/LookupTableImportV2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_1_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0>^string_lookup_1_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::2~
=string_lookup_1_index_table_table_restore/LookupTableImportV2=string_lookup_1_index_table_table_restore/LookupTableImportV2:W S
#
_output_shapes
:?????????
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
Ɲ
?
!__inference__traced_restore_15666
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate
assignvariableop_5_mean
assignvariableop_6_variance
assignvariableop_7_count.
*assignvariableop_8_sequential_dense_kernel,
(assignvariableop_9_sequential_dense_bias5
1assignvariableop_10_sequential_predictions_kernel3
/assignvariableop_11_sequential_predictions_biasY
Ustring_lookup_index_table_table_restore_lookuptableimportv2_string_lookup_index_table]
Ystring_lookup_1_index_table_table_restore_lookuptableimportv2_string_lookup_1_index_table]
Ystring_lookup_2_index_table_table_restore_lookuptableimportv2_string_lookup_2_index_table]
Ystring_lookup_3_index_table_table_restore_lookuptableimportv2_string_lookup_3_index_table]
Ystring_lookup_4_index_table_table_restore_lookuptableimportv2_string_lookup_4_index_table]
Ystring_lookup_5_index_table_table_restore_lookuptableimportv2_string_lookup_5_index_table]
Ystring_lookup_6_index_table_table_restore_lookuptableimportv2_string_lookup_6_index_table]
Ystring_lookup_7_index_table_table_restore_lookuptableimportv2_string_lookup_7_index_table
assignvariableop_12_total
assignvariableop_13_count_16
2assignvariableop_14_adam_sequential_dense_kernel_m4
0assignvariableop_15_adam_sequential_dense_bias_m<
8assignvariableop_16_adam_sequential_predictions_kernel_m:
6assignvariableop_17_adam_sequential_predictions_bias_m6
2assignvariableop_18_adam_sequential_dense_kernel_v4
0assignvariableop_19_adam_sequential_dense_bias_v<
8assignvariableop_20_adam_sequential_predictions_kernel_v:
6assignvariableop_21_adam_sequential_predictions_bias_v
identity_23??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?=string_lookup_1_index_table_table_restore/LookupTableImportV2?=string_lookup_2_index_table_table_restore/LookupTableImportV2?=string_lookup_3_index_table_table_restore/LookupTableImportV2?=string_lookup_4_index_table_table_restore/LookupTableImportV2?=string_lookup_5_index_table_table_restore/LookupTableImportV2?=string_lookup_6_index_table_table_restore/LookupTableImportV2?=string_lookup_7_index_table_table_restore/LookupTableImportV2?;string_lookup_index_table_table_restore/LookupTableImportV2?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*?
value?B?'B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-1/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-1/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-2/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-2/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-3/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-3/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-4/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-4/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-5/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-5/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-6/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-6/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-7/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-7/_table/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::*5
dtypes+
)2'										2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOpassignvariableop_5_meanIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpassignvariableop_6_varianceIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_countIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp*assignvariableop_8_sequential_dense_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp(assignvariableop_9_sequential_dense_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp1assignvariableop_10_sequential_predictions_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp/assignvariableop_11_sequential_predictions_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11?
;string_lookup_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ustring_lookup_index_table_table_restore_lookuptableimportv2_string_lookup_index_tableRestoreV2:tensors:12RestoreV2:tensors:13*	
Tin0*

Tout0	*,
_class"
 loc:@string_lookup_index_table*
_output_shapes
 2=
;string_lookup_index_table_table_restore/LookupTableImportV2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_1_index_table_table_restore_lookuptableimportv2_string_lookup_1_index_tableRestoreV2:tensors:14RestoreV2:tensors:15*	
Tin0*

Tout0	*.
_class$
" loc:@string_lookup_1_index_table*
_output_shapes
 2?
=string_lookup_1_index_table_table_restore/LookupTableImportV2?
=string_lookup_2_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_2_index_table_table_restore_lookuptableimportv2_string_lookup_2_index_tableRestoreV2:tensors:16RestoreV2:tensors:17*	
Tin0*

Tout0	*.
_class$
" loc:@string_lookup_2_index_table*
_output_shapes
 2?
=string_lookup_2_index_table_table_restore/LookupTableImportV2?
=string_lookup_3_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_3_index_table_table_restore_lookuptableimportv2_string_lookup_3_index_tableRestoreV2:tensors:18RestoreV2:tensors:19*	
Tin0*

Tout0	*.
_class$
" loc:@string_lookup_3_index_table*
_output_shapes
 2?
=string_lookup_3_index_table_table_restore/LookupTableImportV2?
=string_lookup_4_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_4_index_table_table_restore_lookuptableimportv2_string_lookup_4_index_tableRestoreV2:tensors:20RestoreV2:tensors:21*	
Tin0*

Tout0	*.
_class$
" loc:@string_lookup_4_index_table*
_output_shapes
 2?
=string_lookup_4_index_table_table_restore/LookupTableImportV2?
=string_lookup_5_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_5_index_table_table_restore_lookuptableimportv2_string_lookup_5_index_tableRestoreV2:tensors:22RestoreV2:tensors:23*	
Tin0*

Tout0	*.
_class$
" loc:@string_lookup_5_index_table*
_output_shapes
 2?
=string_lookup_5_index_table_table_restore/LookupTableImportV2?
=string_lookup_6_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_6_index_table_table_restore_lookuptableimportv2_string_lookup_6_index_tableRestoreV2:tensors:24RestoreV2:tensors:25*	
Tin0*

Tout0	*.
_class$
" loc:@string_lookup_6_index_table*
_output_shapes
 2?
=string_lookup_6_index_table_table_restore/LookupTableImportV2?
=string_lookup_7_index_table_table_restore/LookupTableImportV2LookupTableImportV2Ystring_lookup_7_index_table_table_restore_lookuptableimportv2_string_lookup_7_index_tableRestoreV2:tensors:26RestoreV2:tensors:27*	
Tin0*

Tout0	*.
_class$
" loc:@string_lookup_7_index_table*
_output_shapes
 2?
=string_lookup_7_index_table_table_restore/LookupTableImportV2n
Identity_12IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_count_1Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp2assignvariableop_14_adam_sequential_dense_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp0assignvariableop_15_adam_sequential_dense_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp8assignvariableop_16_adam_sequential_predictions_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp6assignvariableop_17_adam_sequential_predictions_bias_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_sequential_dense_kernel_vIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp0assignvariableop_19_adam_sequential_dense_bias_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp8assignvariableop_20_adam_sequential_predictions_kernel_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp6assignvariableop_21_adam_sequential_predictions_bias_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_219
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_22Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp>^string_lookup_1_index_table_table_restore/LookupTableImportV2>^string_lookup_2_index_table_table_restore/LookupTableImportV2>^string_lookup_3_index_table_table_restore/LookupTableImportV2>^string_lookup_4_index_table_table_restore/LookupTableImportV2>^string_lookup_5_index_table_table_restore/LookupTableImportV2>^string_lookup_6_index_table_table_restore/LookupTableImportV2>^string_lookup_7_index_table_table_restore/LookupTableImportV2<^string_lookup_index_table_table_restore/LookupTableImportV2"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_22?
Identity_23IdentityIdentity_22:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9>^string_lookup_1_index_table_table_restore/LookupTableImportV2>^string_lookup_2_index_table_table_restore/LookupTableImportV2>^string_lookup_3_index_table_table_restore/LookupTableImportV2>^string_lookup_4_index_table_table_restore/LookupTableImportV2>^string_lookup_5_index_table_table_restore/LookupTableImportV2>^string_lookup_6_index_table_table_restore/LookupTableImportV2>^string_lookup_7_index_table_table_restore/LookupTableImportV2<^string_lookup_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2
Identity_23"#
identity_23Identity_23:output:0*?
_input_shapes|
z: ::::::::::::::::::::::::::::::2$
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
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92~
=string_lookup_1_index_table_table_restore/LookupTableImportV2=string_lookup_1_index_table_table_restore/LookupTableImportV22~
=string_lookup_2_index_table_table_restore/LookupTableImportV2=string_lookup_2_index_table_table_restore/LookupTableImportV22~
=string_lookup_3_index_table_table_restore/LookupTableImportV2=string_lookup_3_index_table_table_restore/LookupTableImportV22~
=string_lookup_4_index_table_table_restore/LookupTableImportV2=string_lookup_4_index_table_table_restore/LookupTableImportV22~
=string_lookup_5_index_table_table_restore/LookupTableImportV2=string_lookup_5_index_table_table_restore/LookupTableImportV22~
=string_lookup_6_index_table_table_restore/LookupTableImportV2=string_lookup_6_index_table_table_restore/LookupTableImportV22~
=string_lookup_7_index_table_table_restore/LookupTableImportV2=string_lookup_7_index_table_table_restore/LookupTableImportV22z
;string_lookup_index_table_table_restore/LookupTableImportV2;string_lookup_index_table_table_restore/LookupTableImportV2:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:2.
,
_class"
 loc:@string_lookup_index_table:40
.
_class$
" loc:@string_lookup_1_index_table:40
.
_class$
" loc:@string_lookup_2_index_table:40
.
_class$
" loc:@string_lookup_3_index_table:40
.
_class$
" loc:@string_lookup_4_index_table:40
.
_class$
" loc:@string_lookup_5_index_table:40
.
_class$
" loc:@string_lookup_6_index_table:40
.
_class$
" loc:@string_lookup_7_index_table
? 
?
G__inference_functional_3_layer_call_and_return_conditional_losses_13463

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
functional_1_13416
functional_1_13418	
functional_1_13420
functional_1_13422	
functional_1_13424
functional_1_13426	
functional_1_13428
functional_1_13430	
functional_1_13432
functional_1_13434	
functional_1_13436
functional_1_13438	
functional_1_13440
functional_1_13442	
functional_1_13444
functional_1_13446	
functional_1_13448
functional_1_13450
sequential_13453
sequential_13455
sequential_13457
sequential_13459
identity??$functional_1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
$functional_1/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13functional_1_13416functional_1_13418functional_1_13420functional_1_13422functional_1_13424functional_1_13426functional_1_13428functional_1_13430functional_1_13432functional_1_13434functional_1_13436functional_1_13438functional_1_13440functional_1_13442functional_1_13444functional_1_13446functional_1_13448functional_1_13450*+
Tin$
"2 								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_127292&
$functional_1/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall-functional_1/StatefulPartitionedCall:output:0sequential_13453sequential_13455sequential_13457sequential_13459*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_130852$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0%^functional_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::2L
$functional_1/StatefulPartitionedCall$functional_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
G__inference_functional_1_layer_call_and_return_conditional_losses_12365
age
capital_gain
capital_loss
	education
education_num

fnlwgt
hours_per_week
marital_status
native_country

occupation
race
relationship
sex
	workclassI
Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	G
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
identity??6string_lookup/None_lookup_table_find/LookupTableFindV2?8string_lookup_1/None_lookup_table_find/LookupTableFindV2?8string_lookup_2/None_lookup_table_find/LookupTableFindV2?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?8string_lookup_4/None_lookup_table_find/LookupTableFindV2?8string_lookup_5/None_lookup_table_find/LookupTableFindV2?8string_lookup_6/None_lookup_table_find/LookupTableFindV2?8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_7/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handlenative_countryFstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_6/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handlesexFstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_6/None_lookup_table_find/LookupTableFindV2?
8string_lookup_5/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleraceFstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_5/None_lookup_table_find/LookupTableFindV2?
8string_lookup_4/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handlerelationshipFstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_4/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handle
occupationFstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_2/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handlemarital_statusFstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_2/None_lookup_table_find/LookupTableFindV2?
8string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle	educationFstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handle	workclassDstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2?
concatenate/PartitionedCallPartitionedCallagefnlwgteducation_numcapital_gaincapital_losshours_per_week*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_121902
concatenate/PartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSub$concatenate/PartitionedCall:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrt?
normalization/truedivRealDivnormalization/sub:z:0normalization/Sqrt:y:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Maximum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
"category_encoding_1/bincount/ShapeShapeAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_1/bincount/Shape?
"category_encoding_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_1/bincount/Const?
!category_encoding_1/bincount/ProdProd+category_encoding_1/bincount/Shape:output:0+category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_1/bincount/Prod?
&category_encoding_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_1/bincount/Greater/y?
$category_encoding_1/bincount/GreaterGreater*category_encoding_1/bincount/Prod:output:0/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_1/bincount/Greater?
!category_encoding_1/bincount/CastCast(category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_1/bincount/Cast?
$category_encoding_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_1/bincount/Const_1?
 category_encoding_1/bincount/MaxMaxAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/Max?
"category_encoding_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_1/bincount/add/y?
 category_encoding_1/bincount/addAddV2)category_encoding_1/bincount/Max:output:0+category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/add?
 category_encoding_1/bincount/mulMul%category_encoding_1/bincount/Cast:y:0$category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/mul?
&category_encoding_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_1/bincount/minlength?
$category_encoding_1/bincount/MaximumMaximum/category_encoding_1/bincount/minlength:output:0$category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_1/bincount/Maximum?
$category_encoding_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_1/bincount/Const_2?
*category_encoding_1/bincount/DenseBincountDenseBincountAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_1/bincount/Maximum:z:0-category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_1/bincount/DenseBincount?
"category_encoding_2/bincount/ShapeShapeAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_2/bincount/Shape?
"category_encoding_2/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_2/bincount/Const?
!category_encoding_2/bincount/ProdProd+category_encoding_2/bincount/Shape:output:0+category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_2/bincount/Prod?
&category_encoding_2/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_2/bincount/Greater/y?
$category_encoding_2/bincount/GreaterGreater*category_encoding_2/bincount/Prod:output:0/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_2/bincount/Greater?
!category_encoding_2/bincount/CastCast(category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_2/bincount/Cast?
$category_encoding_2/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_2/bincount/Const_1?
 category_encoding_2/bincount/MaxMaxAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/Max?
"category_encoding_2/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_2/bincount/add/y?
 category_encoding_2/bincount/addAddV2)category_encoding_2/bincount/Max:output:0+category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/add?
 category_encoding_2/bincount/mulMul%category_encoding_2/bincount/Cast:y:0$category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/mul?
&category_encoding_2/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R	2(
&category_encoding_2/bincount/minlength?
$category_encoding_2/bincount/MaximumMaximum/category_encoding_2/bincount/minlength:output:0$category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_2/bincount/Maximum?
$category_encoding_2/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_2/bincount/Const_2?
*category_encoding_2/bincount/DenseBincountDenseBincountAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_2/bincount/Maximum:z:0-category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????	*
binary_output(2,
*category_encoding_2/bincount/DenseBincount?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Maximum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
"category_encoding_4/bincount/ShapeShapeAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_4/bincount/Shape?
"category_encoding_4/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_4/bincount/Const?
!category_encoding_4/bincount/ProdProd+category_encoding_4/bincount/Shape:output:0+category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_4/bincount/Prod?
&category_encoding_4/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_4/bincount/Greater/y?
$category_encoding_4/bincount/GreaterGreater*category_encoding_4/bincount/Prod:output:0/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_4/bincount/Greater?
!category_encoding_4/bincount/CastCast(category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_4/bincount/Cast?
$category_encoding_4/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_4/bincount/Const_1?
 category_encoding_4/bincount/MaxMaxAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/Max?
"category_encoding_4/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_4/bincount/add/y?
 category_encoding_4/bincount/addAddV2)category_encoding_4/bincount/Max:output:0+category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/add?
 category_encoding_4/bincount/mulMul%category_encoding_4/bincount/Cast:y:0$category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/mul?
&category_encoding_4/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_4/bincount/minlength?
$category_encoding_4/bincount/MaximumMaximum/category_encoding_4/bincount/minlength:output:0$category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_4/bincount/Maximum?
$category_encoding_4/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_4/bincount/Const_2?
*category_encoding_4/bincount/DenseBincountDenseBincountAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_4/bincount/Maximum:z:0-category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_4/bincount/DenseBincount?
"category_encoding_5/bincount/ShapeShapeAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_5/bincount/Shape?
"category_encoding_5/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_5/bincount/Const?
!category_encoding_5/bincount/ProdProd+category_encoding_5/bincount/Shape:output:0+category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_5/bincount/Prod?
&category_encoding_5/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_5/bincount/Greater/y?
$category_encoding_5/bincount/GreaterGreater*category_encoding_5/bincount/Prod:output:0/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_5/bincount/Greater?
!category_encoding_5/bincount/CastCast(category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_5/bincount/Cast?
$category_encoding_5/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_5/bincount/Const_1?
 category_encoding_5/bincount/MaxMaxAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/Max?
"category_encoding_5/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_5/bincount/add/y?
 category_encoding_5/bincount/addAddV2)category_encoding_5/bincount/Max:output:0+category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/add?
 category_encoding_5/bincount/mulMul%category_encoding_5/bincount/Cast:y:0$category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/mul?
&category_encoding_5/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_5/bincount/minlength?
$category_encoding_5/bincount/MaximumMaximum/category_encoding_5/bincount/minlength:output:0$category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_5/bincount/Maximum?
$category_encoding_5/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_5/bincount/Const_2?
*category_encoding_5/bincount/DenseBincountDenseBincountAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_5/bincount/Maximum:z:0-category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_5/bincount/DenseBincount?
"category_encoding_6/bincount/ShapeShapeAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_6/bincount/Shape?
"category_encoding_6/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_6/bincount/Const?
!category_encoding_6/bincount/ProdProd+category_encoding_6/bincount/Shape:output:0+category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_6/bincount/Prod?
&category_encoding_6/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_6/bincount/Greater/y?
$category_encoding_6/bincount/GreaterGreater*category_encoding_6/bincount/Prod:output:0/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_6/bincount/Greater?
!category_encoding_6/bincount/CastCast(category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_6/bincount/Cast?
$category_encoding_6/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_6/bincount/Const_1?
 category_encoding_6/bincount/MaxMaxAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/Max?
"category_encoding_6/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_6/bincount/add/y?
 category_encoding_6/bincount/addAddV2)category_encoding_6/bincount/Max:output:0+category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/add?
 category_encoding_6/bincount/mulMul%category_encoding_6/bincount/Cast:y:0$category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/mul?
&category_encoding_6/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_6/bincount/minlength?
$category_encoding_6/bincount/MaximumMaximum/category_encoding_6/bincount/minlength:output:0$category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_6/bincount/Maximum?
$category_encoding_6/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_6/bincount/Const_2?
*category_encoding_6/bincount/DenseBincountDenseBincountAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_6/bincount/Maximum:z:0-category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_6/bincount/DenseBincount?
"category_encoding_7/bincount/ShapeShapeAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_7/bincount/Shape?
"category_encoding_7/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_7/bincount/Const?
!category_encoding_7/bincount/ProdProd+category_encoding_7/bincount/Shape:output:0+category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_7/bincount/Prod?
&category_encoding_7/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_7/bincount/Greater/y?
$category_encoding_7/bincount/GreaterGreater*category_encoding_7/bincount/Prod:output:0/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_7/bincount/Greater?
!category_encoding_7/bincount/CastCast(category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_7/bincount/Cast?
$category_encoding_7/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_7/bincount/Const_1?
 category_encoding_7/bincount/MaxMaxAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/Max?
"category_encoding_7/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_7/bincount/add/y?
 category_encoding_7/bincount/addAddV2)category_encoding_7/bincount/Max:output:0+category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/add?
 category_encoding_7/bincount/mulMul%category_encoding_7/bincount/Cast:y:0$category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/mul?
&category_encoding_7/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2(
&category_encoding_7/bincount/minlength?
$category_encoding_7/bincount/MaximumMaximum/category_encoding_7/bincount/minlength:output:0$category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_7/bincount/Maximum?
$category_encoding_7/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_7/bincount/Const_2?
*category_encoding_7/bincount/DenseBincountDenseBincountAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_7/bincount/Maximum:z:0-category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2,
*category_encoding_7/bincount/DenseBincount?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:01category_encoding/bincount/DenseBincount:output:03category_encoding_1/bincount/DenseBincount:output:03category_encoding_2/bincount/DenseBincount:output:03category_encoding_3/bincount/DenseBincount:output:03category_encoding_4/bincount/DenseBincount:output:03category_encoding_5/bincount/DenseBincount:output:03category_encoding_6/bincount/DenseBincount:output:03category_encoding_7/bincount/DenseBincount:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_123482
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:07^string_lookup/None_lookup_table_find/LookupTableFindV29^string_lookup_1/None_lookup_table_find/LookupTableFindV29^string_lookup_2/None_lookup_table_find/LookupTableFindV29^string_lookup_3/None_lookup_table_find/LookupTableFindV29^string_lookup_4/None_lookup_table_find/LookupTableFindV29^string_lookup_5/None_lookup_table_find/LookupTableFindV29^string_lookup_6/None_lookup_table_find/LookupTableFindV29^string_lookup_7/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV22t
8string_lookup_1/None_lookup_table_find/LookupTableFindV28string_lookup_1/None_lookup_table_find/LookupTableFindV22t
8string_lookup_2/None_lookup_table_find/LookupTableFindV28string_lookup_2/None_lookup_table_find/LookupTableFindV22t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV22t
8string_lookup_4/None_lookup_table_find/LookupTableFindV28string_lookup_4/None_lookup_table_find/LookupTableFindV22t
8string_lookup_5/None_lookup_table_find/LookupTableFindV28string_lookup_5/None_lookup_table_find/LookupTableFindV22t
8string_lookup_6/None_lookup_table_find/LookupTableFindV28string_lookup_6/None_lookup_table_find/LookupTableFindV22t
8string_lookup_7/None_lookup_table_find/LookupTableFindV28string_lookup_7/None_lookup_table_find/LookupTableFindV2:L H
'
_output_shapes
:?????????

_user_specified_nameage:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-gain:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-loss:RN
'
_output_shapes
:?????????
#
_user_specified_name	education:VR
'
_output_shapes
:?????????
'
_user_specified_nameeducation-num:OK
'
_output_shapes
:?????????
 
_user_specified_namefnlwgt:WS
'
_output_shapes
:?????????
(
_user_specified_namehours-per-week:WS
'
_output_shapes
:?????????
(
_user_specified_namemarital-status:WS
'
_output_shapes
:?????????
(
_user_specified_namenative-country:S	O
'
_output_shapes
:?????????
$
_user_specified_name
occupation:M
I
'
_output_shapes
:?????????

_user_specified_namerace:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelationship:LH
'
_output_shapes
:?????????

_user_specified_namesex:RN
'
_output_shapes
:?????????
#
_user_specified_name	workclass
?
?
,__inference_functional_1_layer_call_fn_14831

inputs_age
inputs_capital_gain
inputs_capital_loss
inputs_education
inputs_education_num
inputs_fnlwgt
inputs_hours_per_week
inputs_marital_status
inputs_native_country
inputs_occupation
inputs_race
inputs_relationship

inputs_sex
inputs_workclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_ageinputs_capital_gaininputs_capital_lossinputs_educationinputs_education_numinputs_fnlwgtinputs_hours_per_weekinputs_marital_statusinputs_native_countryinputs_occupationinputs_raceinputs_relationship
inputs_sexinputs_workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*+
Tin$
"2 								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_127292
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/age:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-gain:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-loss:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/education:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/education-num:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/fnlwgt:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/hours-per-week:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/marital-status:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/native-country:Z	V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:T
P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/race:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/relationship:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/workclass
?

?
+__inference_concatenate_layer_call_fn_14966
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_121902
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5
??
?
G__inference_functional_1_layer_call_and_return_conditional_losses_14777

inputs_age
inputs_capital_gain
inputs_capital_loss
inputs_education
inputs_education_num
inputs_fnlwgt
inputs_hours_per_week
inputs_marital_status
inputs_native_country
inputs_occupation
inputs_race
inputs_relationship

inputs_sex
inputs_workclassI
Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	G
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
identity??6string_lookup/None_lookup_table_find/LookupTableFindV2?8string_lookup_1/None_lookup_table_find/LookupTableFindV2?8string_lookup_2/None_lookup_table_find/LookupTableFindV2?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?8string_lookup_4/None_lookup_table_find/LookupTableFindV2?8string_lookup_5/None_lookup_table_find/LookupTableFindV2?8string_lookup_6/None_lookup_table_find/LookupTableFindV2?8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_7/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleinputs_native_countryFstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_6/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handle
inputs_sexFstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_6/None_lookup_table_find/LookupTableFindV2?
8string_lookup_5/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleinputs_raceFstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_5/None_lookup_table_find/LookupTableFindV2?
8string_lookup_4/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleinputs_relationshipFstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_4/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputs_occupationFstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_2/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleinputs_marital_statusFstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_2/None_lookup_table_find/LookupTableFindV2?
8string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleinputs_educationFstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleinputs_workclassDstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2
inputs_ageinputs_fnlwgtinputs_education_numinputs_capital_gaininputs_capital_lossinputs_hours_per_week concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate/concat?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSubconcatenate/concat:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrt?
normalization/truedivRealDivnormalization/sub:z:0normalization/Sqrt:y:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Maximum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
"category_encoding_1/bincount/ShapeShapeAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_1/bincount/Shape?
"category_encoding_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_1/bincount/Const?
!category_encoding_1/bincount/ProdProd+category_encoding_1/bincount/Shape:output:0+category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_1/bincount/Prod?
&category_encoding_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_1/bincount/Greater/y?
$category_encoding_1/bincount/GreaterGreater*category_encoding_1/bincount/Prod:output:0/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_1/bincount/Greater?
!category_encoding_1/bincount/CastCast(category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_1/bincount/Cast?
$category_encoding_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_1/bincount/Const_1?
 category_encoding_1/bincount/MaxMaxAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/Max?
"category_encoding_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_1/bincount/add/y?
 category_encoding_1/bincount/addAddV2)category_encoding_1/bincount/Max:output:0+category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/add?
 category_encoding_1/bincount/mulMul%category_encoding_1/bincount/Cast:y:0$category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/mul?
&category_encoding_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_1/bincount/minlength?
$category_encoding_1/bincount/MaximumMaximum/category_encoding_1/bincount/minlength:output:0$category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_1/bincount/Maximum?
$category_encoding_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_1/bincount/Const_2?
*category_encoding_1/bincount/DenseBincountDenseBincountAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_1/bincount/Maximum:z:0-category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_1/bincount/DenseBincount?
"category_encoding_2/bincount/ShapeShapeAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_2/bincount/Shape?
"category_encoding_2/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_2/bincount/Const?
!category_encoding_2/bincount/ProdProd+category_encoding_2/bincount/Shape:output:0+category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_2/bincount/Prod?
&category_encoding_2/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_2/bincount/Greater/y?
$category_encoding_2/bincount/GreaterGreater*category_encoding_2/bincount/Prod:output:0/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_2/bincount/Greater?
!category_encoding_2/bincount/CastCast(category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_2/bincount/Cast?
$category_encoding_2/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_2/bincount/Const_1?
 category_encoding_2/bincount/MaxMaxAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/Max?
"category_encoding_2/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_2/bincount/add/y?
 category_encoding_2/bincount/addAddV2)category_encoding_2/bincount/Max:output:0+category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/add?
 category_encoding_2/bincount/mulMul%category_encoding_2/bincount/Cast:y:0$category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/mul?
&category_encoding_2/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R	2(
&category_encoding_2/bincount/minlength?
$category_encoding_2/bincount/MaximumMaximum/category_encoding_2/bincount/minlength:output:0$category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_2/bincount/Maximum?
$category_encoding_2/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_2/bincount/Const_2?
*category_encoding_2/bincount/DenseBincountDenseBincountAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_2/bincount/Maximum:z:0-category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????	*
binary_output(2,
*category_encoding_2/bincount/DenseBincount?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Maximum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
"category_encoding_4/bincount/ShapeShapeAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_4/bincount/Shape?
"category_encoding_4/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_4/bincount/Const?
!category_encoding_4/bincount/ProdProd+category_encoding_4/bincount/Shape:output:0+category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_4/bincount/Prod?
&category_encoding_4/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_4/bincount/Greater/y?
$category_encoding_4/bincount/GreaterGreater*category_encoding_4/bincount/Prod:output:0/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_4/bincount/Greater?
!category_encoding_4/bincount/CastCast(category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_4/bincount/Cast?
$category_encoding_4/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_4/bincount/Const_1?
 category_encoding_4/bincount/MaxMaxAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/Max?
"category_encoding_4/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_4/bincount/add/y?
 category_encoding_4/bincount/addAddV2)category_encoding_4/bincount/Max:output:0+category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/add?
 category_encoding_4/bincount/mulMul%category_encoding_4/bincount/Cast:y:0$category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/mul?
&category_encoding_4/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_4/bincount/minlength?
$category_encoding_4/bincount/MaximumMaximum/category_encoding_4/bincount/minlength:output:0$category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_4/bincount/Maximum?
$category_encoding_4/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_4/bincount/Const_2?
*category_encoding_4/bincount/DenseBincountDenseBincountAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_4/bincount/Maximum:z:0-category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_4/bincount/DenseBincount?
"category_encoding_5/bincount/ShapeShapeAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_5/bincount/Shape?
"category_encoding_5/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_5/bincount/Const?
!category_encoding_5/bincount/ProdProd+category_encoding_5/bincount/Shape:output:0+category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_5/bincount/Prod?
&category_encoding_5/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_5/bincount/Greater/y?
$category_encoding_5/bincount/GreaterGreater*category_encoding_5/bincount/Prod:output:0/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_5/bincount/Greater?
!category_encoding_5/bincount/CastCast(category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_5/bincount/Cast?
$category_encoding_5/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_5/bincount/Const_1?
 category_encoding_5/bincount/MaxMaxAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/Max?
"category_encoding_5/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_5/bincount/add/y?
 category_encoding_5/bincount/addAddV2)category_encoding_5/bincount/Max:output:0+category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/add?
 category_encoding_5/bincount/mulMul%category_encoding_5/bincount/Cast:y:0$category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/mul?
&category_encoding_5/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_5/bincount/minlength?
$category_encoding_5/bincount/MaximumMaximum/category_encoding_5/bincount/minlength:output:0$category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_5/bincount/Maximum?
$category_encoding_5/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_5/bincount/Const_2?
*category_encoding_5/bincount/DenseBincountDenseBincountAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_5/bincount/Maximum:z:0-category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_5/bincount/DenseBincount?
"category_encoding_6/bincount/ShapeShapeAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_6/bincount/Shape?
"category_encoding_6/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_6/bincount/Const?
!category_encoding_6/bincount/ProdProd+category_encoding_6/bincount/Shape:output:0+category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_6/bincount/Prod?
&category_encoding_6/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_6/bincount/Greater/y?
$category_encoding_6/bincount/GreaterGreater*category_encoding_6/bincount/Prod:output:0/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_6/bincount/Greater?
!category_encoding_6/bincount/CastCast(category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_6/bincount/Cast?
$category_encoding_6/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_6/bincount/Const_1?
 category_encoding_6/bincount/MaxMaxAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/Max?
"category_encoding_6/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_6/bincount/add/y?
 category_encoding_6/bincount/addAddV2)category_encoding_6/bincount/Max:output:0+category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/add?
 category_encoding_6/bincount/mulMul%category_encoding_6/bincount/Cast:y:0$category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/mul?
&category_encoding_6/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_6/bincount/minlength?
$category_encoding_6/bincount/MaximumMaximum/category_encoding_6/bincount/minlength:output:0$category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_6/bincount/Maximum?
$category_encoding_6/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_6/bincount/Const_2?
*category_encoding_6/bincount/DenseBincountDenseBincountAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_6/bincount/Maximum:z:0-category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_6/bincount/DenseBincount?
"category_encoding_7/bincount/ShapeShapeAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_7/bincount/Shape?
"category_encoding_7/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_7/bincount/Const?
!category_encoding_7/bincount/ProdProd+category_encoding_7/bincount/Shape:output:0+category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_7/bincount/Prod?
&category_encoding_7/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_7/bincount/Greater/y?
$category_encoding_7/bincount/GreaterGreater*category_encoding_7/bincount/Prod:output:0/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_7/bincount/Greater?
!category_encoding_7/bincount/CastCast(category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_7/bincount/Cast?
$category_encoding_7/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_7/bincount/Const_1?
 category_encoding_7/bincount/MaxMaxAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/Max?
"category_encoding_7/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_7/bincount/add/y?
 category_encoding_7/bincount/addAddV2)category_encoding_7/bincount/Max:output:0+category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/add?
 category_encoding_7/bincount/mulMul%category_encoding_7/bincount/Cast:y:0$category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/mul?
&category_encoding_7/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2(
&category_encoding_7/bincount/minlength?
$category_encoding_7/bincount/MaximumMaximum/category_encoding_7/bincount/minlength:output:0$category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_7/bincount/Maximum?
$category_encoding_7/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_7/bincount/Const_2?
*category_encoding_7/bincount/DenseBincountDenseBincountAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_7/bincount/Maximum:z:0-category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2,
*category_encoding_7/bincount/DenseBincountx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2normalization/truediv:z:01category_encoding/bincount/DenseBincount:output:03category_encoding_1/bincount/DenseBincount:output:03category_encoding_2/bincount/DenseBincount:output:03category_encoding_3/bincount/DenseBincount:output:03category_encoding_4/bincount/DenseBincount:output:03category_encoding_5/bincount/DenseBincount:output:03category_encoding_6/bincount/DenseBincount:output:03category_encoding_7/bincount/DenseBincount:output:0"concatenate_1/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????|2
concatenate_1/concat?
IdentityIdentityconcatenate_1/concat:output:07^string_lookup/None_lookup_table_find/LookupTableFindV29^string_lookup_1/None_lookup_table_find/LookupTableFindV29^string_lookup_2/None_lookup_table_find/LookupTableFindV29^string_lookup_3/None_lookup_table_find/LookupTableFindV29^string_lookup_4/None_lookup_table_find/LookupTableFindV29^string_lookup_5/None_lookup_table_find/LookupTableFindV29^string_lookup_6/None_lookup_table_find/LookupTableFindV29^string_lookup_7/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV22t
8string_lookup_1/None_lookup_table_find/LookupTableFindV28string_lookup_1/None_lookup_table_find/LookupTableFindV22t
8string_lookup_2/None_lookup_table_find/LookupTableFindV28string_lookup_2/None_lookup_table_find/LookupTableFindV22t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV22t
8string_lookup_4/None_lookup_table_find/LookupTableFindV28string_lookup_4/None_lookup_table_find/LookupTableFindV22t
8string_lookup_5/None_lookup_table_find/LookupTableFindV28string_lookup_5/None_lookup_table_find/LookupTableFindV22t
8string_lookup_6/None_lookup_table_find/LookupTableFindV28string_lookup_6/None_lookup_table_find/LookupTableFindV22t
8string_lookup_7/None_lookup_table_find/LookupTableFindV28string_lookup_7/None_lookup_table_find/LookupTableFindV2:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/age:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-gain:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-loss:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/education:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/education-num:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/fnlwgt:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/hours-per-week:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/marital-status:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/native-country:Z	V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:T
P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/race:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/relationship:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/workclass
?
,
__inference__destroyer_15092
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?	
?
F__inference_concatenate_layer_call_and_return_conditional_losses_12190

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapest
r:?????????:?????????:?????????:?????????:?????????:?????????:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
K
__inference__creator_15082
identity??string_lookup_3_index_table?
string_lookup_3_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_150*
value_dtype0	2
string_lookup_3_index_table?
IdentityIdentity*string_lookup_3_index_table:table_handle:0^string_lookup_3_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 2:
string_lookup_3_index_tablestring_lookup_3_index_table
?
*
__inference_<lambda>_15378
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
,__inference_functional_3_layer_call_fn_13635
age
capital_gain
capital_loss
	education
education_num

fnlwgt
hours_per_week
marital_status
native_country

occupation
race
relationship
sex
	workclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*/
Tin(
&2$								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_3_layer_call_and_return_conditional_losses_135882
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:?????????

_user_specified_nameage:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-gain:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-loss:RN
'
_output_shapes
:?????????
#
_user_specified_name	education:VR
'
_output_shapes
:?????????
'
_user_specified_nameeducation-num:OK
'
_output_shapes
:?????????
 
_user_specified_namefnlwgt:WS
'
_output_shapes
:?????????
(
_user_specified_namehours-per-week:WS
'
_output_shapes
:?????????
(
_user_specified_namemarital-status:WS
'
_output_shapes
:?????????
(
_user_specified_namenative-country:S	O
'
_output_shapes
:?????????
$
_user_specified_name
occupation:M
I
'
_output_shapes
:?????????

_user_specified_namerace:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelationship:LH
'
_output_shapes
:?????????

_user_specified_namesex:RN
'
_output_shapes
:?????????
#
_user_specified_name	workclass
?
?
*__inference_sequential_layer_call_fn_14932

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_130852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????|::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????|
 
_user_specified_nameinputs
?
*
__inference_<lambda>_15388
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
,__inference_functional_1_layer_call_fn_12996
age
capital_gain
capital_loss
	education
education_num

fnlwgt
hours_per_week
marital_status
native_country

occupation
race
relationship
sex
	workclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*+
Tin$
"2 								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_129572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:?????????

_user_specified_nameage:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-gain:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-loss:RN
'
_output_shapes
:?????????
#
_user_specified_name	education:VR
'
_output_shapes
:?????????
'
_user_specified_nameeducation-num:OK
'
_output_shapes
:?????????
 
_user_specified_namefnlwgt:WS
'
_output_shapes
:?????????
(
_user_specified_namehours-per-week:WS
'
_output_shapes
:?????????
(
_user_specified_namemarital-status:WS
'
_output_shapes
:?????????
(
_user_specified_namenative-country:S	O
'
_output_shapes
:?????????
$
_user_specified_name
occupation:M
I
'
_output_shapes
:?????????

_user_specified_namerace:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelationship:LH
'
_output_shapes
:?????????

_user_specified_namesex:RN
'
_output_shapes
:?????????
#
_user_specified_name	workclass
?
?
-__inference_concatenate_1_layer_call_fn_14993
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity?
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_123482
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????	:?????????:?????????:?????????:?????????:?????????,:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/2:QM
'
_output_shapes
:?????????	
"
_user_specified_name
inputs/3:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/4:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/5:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/6:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/7:QM
'
_output_shapes
:?????????,
"
_user_specified_name
inputs/8
?
.
__inference__initializer_15117
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
,
__inference__destroyer_15137
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
.
__inference__initializer_15102
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
??
?
G__inference_functional_1_layer_call_and_return_conditional_losses_12957

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13I
Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	G
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
identity??6string_lookup/None_lookup_table_find/LookupTableFindV2?8string_lookup_1/None_lookup_table_find/LookupTableFindV2?8string_lookup_2/None_lookup_table_find/LookupTableFindV2?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?8string_lookup_4/None_lookup_table_find/LookupTableFindV2?8string_lookup_5/None_lookup_table_find/LookupTableFindV2?8string_lookup_6/None_lookup_table_find/LookupTableFindV2?8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_7/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleinputs_8Fstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_6/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handle	inputs_12Fstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_6/None_lookup_table_find/LookupTableFindV2?
8string_lookup_5/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handle	inputs_10Fstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_5/None_lookup_table_find/LookupTableFindV2?
8string_lookup_4/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handle	inputs_11Fstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_4/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputs_9Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_2/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleinputs_7Fstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_2/None_lookup_table_find/LookupTableFindV2?
8string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleinputs_3Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handle	inputs_13Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2?
concatenate/PartitionedCallPartitionedCallinputsinputs_5inputs_4inputs_1inputs_2inputs_6*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_121902
concatenate/PartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSub$concatenate/PartitionedCall:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrt?
normalization/truedivRealDivnormalization/sub:z:0normalization/Sqrt:y:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Maximum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
"category_encoding_1/bincount/ShapeShapeAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_1/bincount/Shape?
"category_encoding_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_1/bincount/Const?
!category_encoding_1/bincount/ProdProd+category_encoding_1/bincount/Shape:output:0+category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_1/bincount/Prod?
&category_encoding_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_1/bincount/Greater/y?
$category_encoding_1/bincount/GreaterGreater*category_encoding_1/bincount/Prod:output:0/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_1/bincount/Greater?
!category_encoding_1/bincount/CastCast(category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_1/bincount/Cast?
$category_encoding_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_1/bincount/Const_1?
 category_encoding_1/bincount/MaxMaxAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/Max?
"category_encoding_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_1/bincount/add/y?
 category_encoding_1/bincount/addAddV2)category_encoding_1/bincount/Max:output:0+category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/add?
 category_encoding_1/bincount/mulMul%category_encoding_1/bincount/Cast:y:0$category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/mul?
&category_encoding_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_1/bincount/minlength?
$category_encoding_1/bincount/MaximumMaximum/category_encoding_1/bincount/minlength:output:0$category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_1/bincount/Maximum?
$category_encoding_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_1/bincount/Const_2?
*category_encoding_1/bincount/DenseBincountDenseBincountAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_1/bincount/Maximum:z:0-category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_1/bincount/DenseBincount?
"category_encoding_2/bincount/ShapeShapeAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_2/bincount/Shape?
"category_encoding_2/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_2/bincount/Const?
!category_encoding_2/bincount/ProdProd+category_encoding_2/bincount/Shape:output:0+category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_2/bincount/Prod?
&category_encoding_2/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_2/bincount/Greater/y?
$category_encoding_2/bincount/GreaterGreater*category_encoding_2/bincount/Prod:output:0/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_2/bincount/Greater?
!category_encoding_2/bincount/CastCast(category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_2/bincount/Cast?
$category_encoding_2/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_2/bincount/Const_1?
 category_encoding_2/bincount/MaxMaxAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/Max?
"category_encoding_2/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_2/bincount/add/y?
 category_encoding_2/bincount/addAddV2)category_encoding_2/bincount/Max:output:0+category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/add?
 category_encoding_2/bincount/mulMul%category_encoding_2/bincount/Cast:y:0$category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/mul?
&category_encoding_2/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R	2(
&category_encoding_2/bincount/minlength?
$category_encoding_2/bincount/MaximumMaximum/category_encoding_2/bincount/minlength:output:0$category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_2/bincount/Maximum?
$category_encoding_2/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_2/bincount/Const_2?
*category_encoding_2/bincount/DenseBincountDenseBincountAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_2/bincount/Maximum:z:0-category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????	*
binary_output(2,
*category_encoding_2/bincount/DenseBincount?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Maximum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
"category_encoding_4/bincount/ShapeShapeAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_4/bincount/Shape?
"category_encoding_4/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_4/bincount/Const?
!category_encoding_4/bincount/ProdProd+category_encoding_4/bincount/Shape:output:0+category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_4/bincount/Prod?
&category_encoding_4/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_4/bincount/Greater/y?
$category_encoding_4/bincount/GreaterGreater*category_encoding_4/bincount/Prod:output:0/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_4/bincount/Greater?
!category_encoding_4/bincount/CastCast(category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_4/bincount/Cast?
$category_encoding_4/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_4/bincount/Const_1?
 category_encoding_4/bincount/MaxMaxAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/Max?
"category_encoding_4/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_4/bincount/add/y?
 category_encoding_4/bincount/addAddV2)category_encoding_4/bincount/Max:output:0+category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/add?
 category_encoding_4/bincount/mulMul%category_encoding_4/bincount/Cast:y:0$category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/mul?
&category_encoding_4/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_4/bincount/minlength?
$category_encoding_4/bincount/MaximumMaximum/category_encoding_4/bincount/minlength:output:0$category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_4/bincount/Maximum?
$category_encoding_4/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_4/bincount/Const_2?
*category_encoding_4/bincount/DenseBincountDenseBincountAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_4/bincount/Maximum:z:0-category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_4/bincount/DenseBincount?
"category_encoding_5/bincount/ShapeShapeAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_5/bincount/Shape?
"category_encoding_5/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_5/bincount/Const?
!category_encoding_5/bincount/ProdProd+category_encoding_5/bincount/Shape:output:0+category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_5/bincount/Prod?
&category_encoding_5/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_5/bincount/Greater/y?
$category_encoding_5/bincount/GreaterGreater*category_encoding_5/bincount/Prod:output:0/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_5/bincount/Greater?
!category_encoding_5/bincount/CastCast(category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_5/bincount/Cast?
$category_encoding_5/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_5/bincount/Const_1?
 category_encoding_5/bincount/MaxMaxAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/Max?
"category_encoding_5/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_5/bincount/add/y?
 category_encoding_5/bincount/addAddV2)category_encoding_5/bincount/Max:output:0+category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/add?
 category_encoding_5/bincount/mulMul%category_encoding_5/bincount/Cast:y:0$category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/mul?
&category_encoding_5/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_5/bincount/minlength?
$category_encoding_5/bincount/MaximumMaximum/category_encoding_5/bincount/minlength:output:0$category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_5/bincount/Maximum?
$category_encoding_5/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_5/bincount/Const_2?
*category_encoding_5/bincount/DenseBincountDenseBincountAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_5/bincount/Maximum:z:0-category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_5/bincount/DenseBincount?
"category_encoding_6/bincount/ShapeShapeAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_6/bincount/Shape?
"category_encoding_6/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_6/bincount/Const?
!category_encoding_6/bincount/ProdProd+category_encoding_6/bincount/Shape:output:0+category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_6/bincount/Prod?
&category_encoding_6/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_6/bincount/Greater/y?
$category_encoding_6/bincount/GreaterGreater*category_encoding_6/bincount/Prod:output:0/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_6/bincount/Greater?
!category_encoding_6/bincount/CastCast(category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_6/bincount/Cast?
$category_encoding_6/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_6/bincount/Const_1?
 category_encoding_6/bincount/MaxMaxAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/Max?
"category_encoding_6/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_6/bincount/add/y?
 category_encoding_6/bincount/addAddV2)category_encoding_6/bincount/Max:output:0+category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/add?
 category_encoding_6/bincount/mulMul%category_encoding_6/bincount/Cast:y:0$category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/mul?
&category_encoding_6/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_6/bincount/minlength?
$category_encoding_6/bincount/MaximumMaximum/category_encoding_6/bincount/minlength:output:0$category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_6/bincount/Maximum?
$category_encoding_6/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_6/bincount/Const_2?
*category_encoding_6/bincount/DenseBincountDenseBincountAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_6/bincount/Maximum:z:0-category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_6/bincount/DenseBincount?
"category_encoding_7/bincount/ShapeShapeAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_7/bincount/Shape?
"category_encoding_7/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_7/bincount/Const?
!category_encoding_7/bincount/ProdProd+category_encoding_7/bincount/Shape:output:0+category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_7/bincount/Prod?
&category_encoding_7/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_7/bincount/Greater/y?
$category_encoding_7/bincount/GreaterGreater*category_encoding_7/bincount/Prod:output:0/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_7/bincount/Greater?
!category_encoding_7/bincount/CastCast(category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_7/bincount/Cast?
$category_encoding_7/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_7/bincount/Const_1?
 category_encoding_7/bincount/MaxMaxAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/Max?
"category_encoding_7/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_7/bincount/add/y?
 category_encoding_7/bincount/addAddV2)category_encoding_7/bincount/Max:output:0+category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/add?
 category_encoding_7/bincount/mulMul%category_encoding_7/bincount/Cast:y:0$category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/mul?
&category_encoding_7/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2(
&category_encoding_7/bincount/minlength?
$category_encoding_7/bincount/MaximumMaximum/category_encoding_7/bincount/minlength:output:0$category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_7/bincount/Maximum?
$category_encoding_7/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_7/bincount/Const_2?
*category_encoding_7/bincount/DenseBincountDenseBincountAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_7/bincount/Maximum:z:0-category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2,
*category_encoding_7/bincount/DenseBincount?
concatenate_1/PartitionedCallPartitionedCallnormalization/truediv:z:01category_encoding/bincount/DenseBincount:output:03category_encoding_1/bincount/DenseBincount:output:03category_encoding_2/bincount/DenseBincount:output:03category_encoding_3/bincount/DenseBincount:output:03category_encoding_4/bincount/DenseBincount:output:03category_encoding_5/bincount/DenseBincount:output:03category_encoding_6/bincount/DenseBincount:output:03category_encoding_7/bincount/DenseBincount:output:0*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_concatenate_1_layer_call_and_return_conditional_losses_123482
concatenate_1/PartitionedCall?
IdentityIdentity&concatenate_1/PartitionedCall:output:07^string_lookup/None_lookup_table_find/LookupTableFindV29^string_lookup_1/None_lookup_table_find/LookupTableFindV29^string_lookup_2/None_lookup_table_find/LookupTableFindV29^string_lookup_3/None_lookup_table_find/LookupTableFindV29^string_lookup_4/None_lookup_table_find/LookupTableFindV29^string_lookup_5/None_lookup_table_find/LookupTableFindV29^string_lookup_6/None_lookup_table_find/LookupTableFindV29^string_lookup_7/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV22t
8string_lookup_1/None_lookup_table_find/LookupTableFindV28string_lookup_1/None_lookup_table_find/LookupTableFindV22t
8string_lookup_2/None_lookup_table_find/LookupTableFindV28string_lookup_2/None_lookup_table_find/LookupTableFindV22t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV22t
8string_lookup_4/None_lookup_table_find/LookupTableFindV28string_lookup_4/None_lookup_table_find/LookupTableFindV22t
8string_lookup_5/None_lookup_table_find/LookupTableFindV28string_lookup_5/None_lookup_table_find/LookupTableFindV22t
8string_lookup_6/None_lookup_table_find/LookupTableFindV28string_lookup_6/None_lookup_table_find/LookupTableFindV22t
8string_lookup_7/None_lookup_table_find/LookupTableFindV28string_lookup_7/None_lookup_table_find/LookupTableFindV2:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_functional_3_layer_call_fn_14363

inputs_age
inputs_capital_gain
inputs_capital_loss
inputs_education
inputs_education_num
inputs_fnlwgt
inputs_hours_per_week
inputs_marital_status
inputs_native_country
inputs_occupation
inputs_race
inputs_relationship

inputs_sex
inputs_workclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCall
inputs_ageinputs_capital_gaininputs_capital_lossinputs_educationinputs_education_numinputs_fnlwgtinputs_hours_per_weekinputs_marital_statusinputs_native_countryinputs_occupationinputs_raceinputs_relationship
inputs_sexinputs_workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*/
Tin(
&2$								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_3_layer_call_and_return_conditional_losses_134632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/age:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-gain:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-loss:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/education:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/education-num:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/fnlwgt:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/hours-per-week:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/marital-status:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/native-country:Z	V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:T
P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/race:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/relationship:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/workclass
?
.
__inference__initializer_15057
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
K
__inference__creator_15052
identity??string_lookup_1_index_table?
string_lookup_1_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name
table_86*
value_dtype0	2
string_lookup_1_index_table?
IdentityIdentity*string_lookup_1_index_table:table_handle:0^string_lookup_1_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 2:
string_lookup_1_index_tablestring_lookup_1_index_table
?
,
__inference__destroyer_15047
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
K
__inference__creator_15127
identity??string_lookup_6_index_table?
string_lookup_6_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_246*
value_dtype0	2
string_lookup_6_index_table?
IdentityIdentity*string_lookup_6_index_table:table_handle:0^string_lookup_6_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 2:
string_lookup_6_index_tablestring_lookup_6_index_table
?	
?
__inference_restore_fn_15341
restored_tensors_0
restored_tensors_1	N
Jstring_lookup_6_index_table_table_restore_lookuptableimportv2_table_handle
identity??=string_lookup_6_index_table_table_restore/LookupTableImportV2?
=string_lookup_6_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_6_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2?
=string_lookup_6_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0>^string_lookup_6_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::2~
=string_lookup_6_index_table_table_restore/LookupTableImportV2=string_lookup_6_index_table_table_restore/LookupTableImportV2:W S
#
_output_shapes
:?????????
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
? 
?
G__inference_functional_3_layer_call_and_return_conditional_losses_13588

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
inputs_9
	inputs_10
	inputs_11
	inputs_12
	inputs_13
functional_1_13541
functional_1_13543	
functional_1_13545
functional_1_13547	
functional_1_13549
functional_1_13551	
functional_1_13553
functional_1_13555	
functional_1_13557
functional_1_13559	
functional_1_13561
functional_1_13563	
functional_1_13565
functional_1_13567	
functional_1_13569
functional_1_13571	
functional_1_13573
functional_1_13575
sequential_13578
sequential_13580
sequential_13582
sequential_13584
identity??$functional_1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
$functional_1/StatefulPartitionedCallStatefulPartitionedCallinputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8inputs_9	inputs_10	inputs_11	inputs_12	inputs_13functional_1_13541functional_1_13543functional_1_13545functional_1_13547functional_1_13549functional_1_13551functional_1_13553functional_1_13555functional_1_13557functional_1_13559functional_1_13561functional_1_13563functional_1_13565functional_1_13567functional_1_13569functional_1_13571functional_1_13573functional_1_13575*+
Tin$
"2 								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_129572&
$functional_1/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall-functional_1/StatefulPartitionedCall:output:0sequential_13578sequential_13580sequential_13582sequential_13584*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_131122$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0%^functional_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::2L
$functional_1/StatefulPartitionedCall$functional_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O	K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:O
K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
,
__inference__destroyer_15107
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
??
?
 __inference__wrapped_model_12138
age
capital_gain
capital_loss
	education
education_num

fnlwgt
hours_per_week
marital_status
native_country

occupation
race
relationship
sex
	workclassc
_functional_3_functional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handled
`functional_3_functional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value	c
_functional_3_functional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handled
`functional_3_functional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value	c
_functional_3_functional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handled
`functional_3_functional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value	c
_functional_3_functional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handled
`functional_3_functional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value	c
_functional_3_functional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handled
`functional_3_functional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	c
_functional_3_functional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handled
`functional_3_functional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value	c
_functional_3_functional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handled
`functional_3_functional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	a
]functional_3_functional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handleb
^functional_3_functional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value	K
Gfunctional_3_functional_1_normalization_reshape_readvariableop_resourceM
Ifunctional_3_functional_1_normalization_reshape_1_readvariableop_resource@
<functional_3_sequential_dense_matmul_readvariableop_resourceA
=functional_3_sequential_dense_biasadd_readvariableop_resourceF
Bfunctional_3_sequential_predictions_matmul_readvariableop_resourceG
Cfunctional_3_sequential_predictions_biasadd_readvariableop_resource
identity??Pfunctional_3/functional_1/string_lookup/None_lookup_table_find/LookupTableFindV2?Rfunctional_3/functional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?Rfunctional_3/functional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2?Rfunctional_3/functional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2?Rfunctional_3/functional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2?Rfunctional_3/functional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2?Rfunctional_3/functional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2?Rfunctional_3/functional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2?
Rfunctional_3/functional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2LookupTableFindV2_functional_3_functional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handlenative_country`functional_3_functional_1_string_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2T
Rfunctional_3/functional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2?
Rfunctional_3/functional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2LookupTableFindV2_functional_3_functional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handlesex`functional_3_functional_1_string_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2T
Rfunctional_3/functional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2?
Rfunctional_3/functional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2LookupTableFindV2_functional_3_functional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handlerace`functional_3_functional_1_string_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2T
Rfunctional_3/functional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2?
Rfunctional_3/functional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2LookupTableFindV2_functional_3_functional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handlerelationship`functional_3_functional_1_string_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2T
Rfunctional_3/functional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2?
Rfunctional_3/functional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2_functional_3_functional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handle
occupation`functional_3_functional_1_string_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2T
Rfunctional_3/functional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2?
Rfunctional_3/functional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2LookupTableFindV2_functional_3_functional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handlemarital_status`functional_3_functional_1_string_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2T
Rfunctional_3/functional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2?
Rfunctional_3/functional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2_functional_3_functional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handle	education`functional_3_functional_1_string_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2T
Rfunctional_3/functional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2?
Pfunctional_3/functional_1/string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2]functional_3_functional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_table_handle	workclass^functional_3_functional_1_string_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2R
Pfunctional_3/functional_1/string_lookup/None_lookup_table_find/LookupTableFindV2?
1functional_3/functional_1/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :23
1functional_3/functional_1/concatenate/concat/axis?
,functional_3/functional_1/concatenate/concatConcatV2agefnlwgteducation_numcapital_gaincapital_losshours_per_week:functional_3/functional_1/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2.
,functional_3/functional_1/concatenate/concat?
>functional_3/functional_1/normalization/Reshape/ReadVariableOpReadVariableOpGfunctional_3_functional_1_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02@
>functional_3/functional_1/normalization/Reshape/ReadVariableOp?
5functional_3/functional_1/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      27
5functional_3/functional_1/normalization/Reshape/shape?
/functional_3/functional_1/normalization/ReshapeReshapeFfunctional_3/functional_1/normalization/Reshape/ReadVariableOp:value:0>functional_3/functional_1/normalization/Reshape/shape:output:0*
T0*
_output_shapes

:21
/functional_3/functional_1/normalization/Reshape?
@functional_3/functional_1/normalization/Reshape_1/ReadVariableOpReadVariableOpIfunctional_3_functional_1_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02B
@functional_3/functional_1/normalization/Reshape_1/ReadVariableOp?
7functional_3/functional_1/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      29
7functional_3/functional_1/normalization/Reshape_1/shape?
1functional_3/functional_1/normalization/Reshape_1ReshapeHfunctional_3/functional_1/normalization/Reshape_1/ReadVariableOp:value:0@functional_3/functional_1/normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:23
1functional_3/functional_1/normalization/Reshape_1?
+functional_3/functional_1/normalization/subSub5functional_3/functional_1/concatenate/concat:output:08functional_3/functional_1/normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2-
+functional_3/functional_1/normalization/sub?
,functional_3/functional_1/normalization/SqrtSqrt:functional_3/functional_1/normalization/Reshape_1:output:0*
T0*
_output_shapes

:2.
,functional_3/functional_1/normalization/Sqrt?
/functional_3/functional_1/normalization/truedivRealDiv/functional_3/functional_1/normalization/sub:z:00functional_3/functional_1/normalization/Sqrt:y:0*
T0*'
_output_shapes
:?????????21
/functional_3/functional_1/normalization/truediv?
:functional_3/functional_1/category_encoding/bincount/ShapeShapeYfunctional_3/functional_1/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2<
:functional_3/functional_1/category_encoding/bincount/Shape?
:functional_3/functional_1/category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2<
:functional_3/functional_1/category_encoding/bincount/Const?
9functional_3/functional_1/category_encoding/bincount/ProdProdCfunctional_3/functional_1/category_encoding/bincount/Shape:output:0Cfunctional_3/functional_1/category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2;
9functional_3/functional_1/category_encoding/bincount/Prod?
>functional_3/functional_1/category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2@
>functional_3/functional_1/category_encoding/bincount/Greater/y?
<functional_3/functional_1/category_encoding/bincount/GreaterGreaterBfunctional_3/functional_1/category_encoding/bincount/Prod:output:0Gfunctional_3/functional_1/category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2>
<functional_3/functional_1/category_encoding/bincount/Greater?
9functional_3/functional_1/category_encoding/bincount/CastCast@functional_3/functional_1/category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2;
9functional_3/functional_1/category_encoding/bincount/Cast?
<functional_3/functional_1/category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2>
<functional_3/functional_1/category_encoding/bincount/Const_1?
8functional_3/functional_1/category_encoding/bincount/MaxMaxYfunctional_3/functional_1/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0Efunctional_3/functional_1/category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2:
8functional_3/functional_1/category_encoding/bincount/Max?
:functional_3/functional_1/category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2<
:functional_3/functional_1/category_encoding/bincount/add/y?
8functional_3/functional_1/category_encoding/bincount/addAddV2Afunctional_3/functional_1/category_encoding/bincount/Max:output:0Cfunctional_3/functional_1/category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2:
8functional_3/functional_1/category_encoding/bincount/add?
8functional_3/functional_1/category_encoding/bincount/mulMul=functional_3/functional_1/category_encoding/bincount/Cast:y:0<functional_3/functional_1/category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2:
8functional_3/functional_1/category_encoding/bincount/mul?
>functional_3/functional_1/category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2@
>functional_3/functional_1/category_encoding/bincount/minlength?
<functional_3/functional_1/category_encoding/bincount/MaximumMaximumGfunctional_3/functional_1/category_encoding/bincount/minlength:output:0<functional_3/functional_1/category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2>
<functional_3/functional_1/category_encoding/bincount/Maximum?
<functional_3/functional_1/category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2>
<functional_3/functional_1/category_encoding/bincount/Const_2?
Bfunctional_3/functional_1/category_encoding/bincount/DenseBincountDenseBincountYfunctional_3/functional_1/string_lookup/None_lookup_table_find/LookupTableFindV2:values:0@functional_3/functional_1/category_encoding/bincount/Maximum:z:0Efunctional_3/functional_1/category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2D
Bfunctional_3/functional_1/category_encoding/bincount/DenseBincount?
<functional_3/functional_1/category_encoding_1/bincount/ShapeShape[functional_3/functional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2>
<functional_3/functional_1/category_encoding_1/bincount/Shape?
<functional_3/functional_1/category_encoding_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<functional_3/functional_1/category_encoding_1/bincount/Const?
;functional_3/functional_1/category_encoding_1/bincount/ProdProdEfunctional_3/functional_1/category_encoding_1/bincount/Shape:output:0Efunctional_3/functional_1/category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_1/bincount/Prod?
@functional_3/functional_1/category_encoding_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2B
@functional_3/functional_1/category_encoding_1/bincount/Greater/y?
>functional_3/functional_1/category_encoding_1/bincount/GreaterGreaterDfunctional_3/functional_1/category_encoding_1/bincount/Prod:output:0Ifunctional_3/functional_1/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_1/bincount/Greater?
;functional_3/functional_1/category_encoding_1/bincount/CastCastBfunctional_3/functional_1/category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_1/bincount/Cast?
>functional_3/functional_1/category_encoding_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2@
>functional_3/functional_1/category_encoding_1/bincount/Const_1?
:functional_3/functional_1/category_encoding_1/bincount/MaxMax[functional_3/functional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0Gfunctional_3/functional_1/category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_1/bincount/Max?
<functional_3/functional_1/category_encoding_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2>
<functional_3/functional_1/category_encoding_1/bincount/add/y?
:functional_3/functional_1/category_encoding_1/bincount/addAddV2Cfunctional_3/functional_1/category_encoding_1/bincount/Max:output:0Efunctional_3/functional_1/category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_1/bincount/add?
:functional_3/functional_1/category_encoding_1/bincount/mulMul?functional_3/functional_1/category_encoding_1/bincount/Cast:y:0>functional_3/functional_1/category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_1/bincount/mul?
@functional_3/functional_1/category_encoding_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2B
@functional_3/functional_1/category_encoding_1/bincount/minlength?
>functional_3/functional_1/category_encoding_1/bincount/MaximumMaximumIfunctional_3/functional_1/category_encoding_1/bincount/minlength:output:0>functional_3/functional_1/category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_1/bincount/Maximum?
>functional_3/functional_1/category_encoding_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2@
>functional_3/functional_1/category_encoding_1/bincount/Const_2?
Dfunctional_3/functional_1/category_encoding_1/bincount/DenseBincountDenseBincount[functional_3/functional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0Bfunctional_3/functional_1/category_encoding_1/bincount/Maximum:z:0Gfunctional_3/functional_1/category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2F
Dfunctional_3/functional_1/category_encoding_1/bincount/DenseBincount?
<functional_3/functional_1/category_encoding_2/bincount/ShapeShape[functional_3/functional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2>
<functional_3/functional_1/category_encoding_2/bincount/Shape?
<functional_3/functional_1/category_encoding_2/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<functional_3/functional_1/category_encoding_2/bincount/Const?
;functional_3/functional_1/category_encoding_2/bincount/ProdProdEfunctional_3/functional_1/category_encoding_2/bincount/Shape:output:0Efunctional_3/functional_1/category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_2/bincount/Prod?
@functional_3/functional_1/category_encoding_2/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2B
@functional_3/functional_1/category_encoding_2/bincount/Greater/y?
>functional_3/functional_1/category_encoding_2/bincount/GreaterGreaterDfunctional_3/functional_1/category_encoding_2/bincount/Prod:output:0Ifunctional_3/functional_1/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_2/bincount/Greater?
;functional_3/functional_1/category_encoding_2/bincount/CastCastBfunctional_3/functional_1/category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_2/bincount/Cast?
>functional_3/functional_1/category_encoding_2/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2@
>functional_3/functional_1/category_encoding_2/bincount/Const_1?
:functional_3/functional_1/category_encoding_2/bincount/MaxMax[functional_3/functional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0Gfunctional_3/functional_1/category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_2/bincount/Max?
<functional_3/functional_1/category_encoding_2/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2>
<functional_3/functional_1/category_encoding_2/bincount/add/y?
:functional_3/functional_1/category_encoding_2/bincount/addAddV2Cfunctional_3/functional_1/category_encoding_2/bincount/Max:output:0Efunctional_3/functional_1/category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_2/bincount/add?
:functional_3/functional_1/category_encoding_2/bincount/mulMul?functional_3/functional_1/category_encoding_2/bincount/Cast:y:0>functional_3/functional_1/category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_2/bincount/mul?
@functional_3/functional_1/category_encoding_2/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R	2B
@functional_3/functional_1/category_encoding_2/bincount/minlength?
>functional_3/functional_1/category_encoding_2/bincount/MaximumMaximumIfunctional_3/functional_1/category_encoding_2/bincount/minlength:output:0>functional_3/functional_1/category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_2/bincount/Maximum?
>functional_3/functional_1/category_encoding_2/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2@
>functional_3/functional_1/category_encoding_2/bincount/Const_2?
Dfunctional_3/functional_1/category_encoding_2/bincount/DenseBincountDenseBincount[functional_3/functional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0Bfunctional_3/functional_1/category_encoding_2/bincount/Maximum:z:0Gfunctional_3/functional_1/category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????	*
binary_output(2F
Dfunctional_3/functional_1/category_encoding_2/bincount/DenseBincount?
<functional_3/functional_1/category_encoding_3/bincount/ShapeShape[functional_3/functional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2>
<functional_3/functional_1/category_encoding_3/bincount/Shape?
<functional_3/functional_1/category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<functional_3/functional_1/category_encoding_3/bincount/Const?
;functional_3/functional_1/category_encoding_3/bincount/ProdProdEfunctional_3/functional_1/category_encoding_3/bincount/Shape:output:0Efunctional_3/functional_1/category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_3/bincount/Prod?
@functional_3/functional_1/category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2B
@functional_3/functional_1/category_encoding_3/bincount/Greater/y?
>functional_3/functional_1/category_encoding_3/bincount/GreaterGreaterDfunctional_3/functional_1/category_encoding_3/bincount/Prod:output:0Ifunctional_3/functional_1/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_3/bincount/Greater?
;functional_3/functional_1/category_encoding_3/bincount/CastCastBfunctional_3/functional_1/category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_3/bincount/Cast?
>functional_3/functional_1/category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2@
>functional_3/functional_1/category_encoding_3/bincount/Const_1?
:functional_3/functional_1/category_encoding_3/bincount/MaxMax[functional_3/functional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0Gfunctional_3/functional_1/category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_3/bincount/Max?
<functional_3/functional_1/category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2>
<functional_3/functional_1/category_encoding_3/bincount/add/y?
:functional_3/functional_1/category_encoding_3/bincount/addAddV2Cfunctional_3/functional_1/category_encoding_3/bincount/Max:output:0Efunctional_3/functional_1/category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_3/bincount/add?
:functional_3/functional_1/category_encoding_3/bincount/mulMul?functional_3/functional_1/category_encoding_3/bincount/Cast:y:0>functional_3/functional_1/category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_3/bincount/mul?
@functional_3/functional_1/category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2B
@functional_3/functional_1/category_encoding_3/bincount/minlength?
>functional_3/functional_1/category_encoding_3/bincount/MaximumMaximumIfunctional_3/functional_1/category_encoding_3/bincount/minlength:output:0>functional_3/functional_1/category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_3/bincount/Maximum?
>functional_3/functional_1/category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2@
>functional_3/functional_1/category_encoding_3/bincount/Const_2?
Dfunctional_3/functional_1/category_encoding_3/bincount/DenseBincountDenseBincount[functional_3/functional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0Bfunctional_3/functional_1/category_encoding_3/bincount/Maximum:z:0Gfunctional_3/functional_1/category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2F
Dfunctional_3/functional_1/category_encoding_3/bincount/DenseBincount?
<functional_3/functional_1/category_encoding_4/bincount/ShapeShape[functional_3/functional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2>
<functional_3/functional_1/category_encoding_4/bincount/Shape?
<functional_3/functional_1/category_encoding_4/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<functional_3/functional_1/category_encoding_4/bincount/Const?
;functional_3/functional_1/category_encoding_4/bincount/ProdProdEfunctional_3/functional_1/category_encoding_4/bincount/Shape:output:0Efunctional_3/functional_1/category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_4/bincount/Prod?
@functional_3/functional_1/category_encoding_4/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2B
@functional_3/functional_1/category_encoding_4/bincount/Greater/y?
>functional_3/functional_1/category_encoding_4/bincount/GreaterGreaterDfunctional_3/functional_1/category_encoding_4/bincount/Prod:output:0Ifunctional_3/functional_1/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_4/bincount/Greater?
;functional_3/functional_1/category_encoding_4/bincount/CastCastBfunctional_3/functional_1/category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_4/bincount/Cast?
>functional_3/functional_1/category_encoding_4/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2@
>functional_3/functional_1/category_encoding_4/bincount/Const_1?
:functional_3/functional_1/category_encoding_4/bincount/MaxMax[functional_3/functional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0Gfunctional_3/functional_1/category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_4/bincount/Max?
<functional_3/functional_1/category_encoding_4/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2>
<functional_3/functional_1/category_encoding_4/bincount/add/y?
:functional_3/functional_1/category_encoding_4/bincount/addAddV2Cfunctional_3/functional_1/category_encoding_4/bincount/Max:output:0Efunctional_3/functional_1/category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_4/bincount/add?
:functional_3/functional_1/category_encoding_4/bincount/mulMul?functional_3/functional_1/category_encoding_4/bincount/Cast:y:0>functional_3/functional_1/category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_4/bincount/mul?
@functional_3/functional_1/category_encoding_4/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2B
@functional_3/functional_1/category_encoding_4/bincount/minlength?
>functional_3/functional_1/category_encoding_4/bincount/MaximumMaximumIfunctional_3/functional_1/category_encoding_4/bincount/minlength:output:0>functional_3/functional_1/category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_4/bincount/Maximum?
>functional_3/functional_1/category_encoding_4/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2@
>functional_3/functional_1/category_encoding_4/bincount/Const_2?
Dfunctional_3/functional_1/category_encoding_4/bincount/DenseBincountDenseBincount[functional_3/functional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0Bfunctional_3/functional_1/category_encoding_4/bincount/Maximum:z:0Gfunctional_3/functional_1/category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2F
Dfunctional_3/functional_1/category_encoding_4/bincount/DenseBincount?
<functional_3/functional_1/category_encoding_5/bincount/ShapeShape[functional_3/functional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2>
<functional_3/functional_1/category_encoding_5/bincount/Shape?
<functional_3/functional_1/category_encoding_5/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<functional_3/functional_1/category_encoding_5/bincount/Const?
;functional_3/functional_1/category_encoding_5/bincount/ProdProdEfunctional_3/functional_1/category_encoding_5/bincount/Shape:output:0Efunctional_3/functional_1/category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_5/bincount/Prod?
@functional_3/functional_1/category_encoding_5/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2B
@functional_3/functional_1/category_encoding_5/bincount/Greater/y?
>functional_3/functional_1/category_encoding_5/bincount/GreaterGreaterDfunctional_3/functional_1/category_encoding_5/bincount/Prod:output:0Ifunctional_3/functional_1/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_5/bincount/Greater?
;functional_3/functional_1/category_encoding_5/bincount/CastCastBfunctional_3/functional_1/category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_5/bincount/Cast?
>functional_3/functional_1/category_encoding_5/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2@
>functional_3/functional_1/category_encoding_5/bincount/Const_1?
:functional_3/functional_1/category_encoding_5/bincount/MaxMax[functional_3/functional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0Gfunctional_3/functional_1/category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_5/bincount/Max?
<functional_3/functional_1/category_encoding_5/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2>
<functional_3/functional_1/category_encoding_5/bincount/add/y?
:functional_3/functional_1/category_encoding_5/bincount/addAddV2Cfunctional_3/functional_1/category_encoding_5/bincount/Max:output:0Efunctional_3/functional_1/category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_5/bincount/add?
:functional_3/functional_1/category_encoding_5/bincount/mulMul?functional_3/functional_1/category_encoding_5/bincount/Cast:y:0>functional_3/functional_1/category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_5/bincount/mul?
@functional_3/functional_1/category_encoding_5/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2B
@functional_3/functional_1/category_encoding_5/bincount/minlength?
>functional_3/functional_1/category_encoding_5/bincount/MaximumMaximumIfunctional_3/functional_1/category_encoding_5/bincount/minlength:output:0>functional_3/functional_1/category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_5/bincount/Maximum?
>functional_3/functional_1/category_encoding_5/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2@
>functional_3/functional_1/category_encoding_5/bincount/Const_2?
Dfunctional_3/functional_1/category_encoding_5/bincount/DenseBincountDenseBincount[functional_3/functional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0Bfunctional_3/functional_1/category_encoding_5/bincount/Maximum:z:0Gfunctional_3/functional_1/category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2F
Dfunctional_3/functional_1/category_encoding_5/bincount/DenseBincount?
<functional_3/functional_1/category_encoding_6/bincount/ShapeShape[functional_3/functional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2>
<functional_3/functional_1/category_encoding_6/bincount/Shape?
<functional_3/functional_1/category_encoding_6/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<functional_3/functional_1/category_encoding_6/bincount/Const?
;functional_3/functional_1/category_encoding_6/bincount/ProdProdEfunctional_3/functional_1/category_encoding_6/bincount/Shape:output:0Efunctional_3/functional_1/category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_6/bincount/Prod?
@functional_3/functional_1/category_encoding_6/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2B
@functional_3/functional_1/category_encoding_6/bincount/Greater/y?
>functional_3/functional_1/category_encoding_6/bincount/GreaterGreaterDfunctional_3/functional_1/category_encoding_6/bincount/Prod:output:0Ifunctional_3/functional_1/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_6/bincount/Greater?
;functional_3/functional_1/category_encoding_6/bincount/CastCastBfunctional_3/functional_1/category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_6/bincount/Cast?
>functional_3/functional_1/category_encoding_6/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2@
>functional_3/functional_1/category_encoding_6/bincount/Const_1?
:functional_3/functional_1/category_encoding_6/bincount/MaxMax[functional_3/functional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0Gfunctional_3/functional_1/category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_6/bincount/Max?
<functional_3/functional_1/category_encoding_6/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2>
<functional_3/functional_1/category_encoding_6/bincount/add/y?
:functional_3/functional_1/category_encoding_6/bincount/addAddV2Cfunctional_3/functional_1/category_encoding_6/bincount/Max:output:0Efunctional_3/functional_1/category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_6/bincount/add?
:functional_3/functional_1/category_encoding_6/bincount/mulMul?functional_3/functional_1/category_encoding_6/bincount/Cast:y:0>functional_3/functional_1/category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_6/bincount/mul?
@functional_3/functional_1/category_encoding_6/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2B
@functional_3/functional_1/category_encoding_6/bincount/minlength?
>functional_3/functional_1/category_encoding_6/bincount/MaximumMaximumIfunctional_3/functional_1/category_encoding_6/bincount/minlength:output:0>functional_3/functional_1/category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_6/bincount/Maximum?
>functional_3/functional_1/category_encoding_6/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2@
>functional_3/functional_1/category_encoding_6/bincount/Const_2?
Dfunctional_3/functional_1/category_encoding_6/bincount/DenseBincountDenseBincount[functional_3/functional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0Bfunctional_3/functional_1/category_encoding_6/bincount/Maximum:z:0Gfunctional_3/functional_1/category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2F
Dfunctional_3/functional_1/category_encoding_6/bincount/DenseBincount?
<functional_3/functional_1/category_encoding_7/bincount/ShapeShape[functional_3/functional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2>
<functional_3/functional_1/category_encoding_7/bincount/Shape?
<functional_3/functional_1/category_encoding_7/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2>
<functional_3/functional_1/category_encoding_7/bincount/Const?
;functional_3/functional_1/category_encoding_7/bincount/ProdProdEfunctional_3/functional_1/category_encoding_7/bincount/Shape:output:0Efunctional_3/functional_1/category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_7/bincount/Prod?
@functional_3/functional_1/category_encoding_7/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2B
@functional_3/functional_1/category_encoding_7/bincount/Greater/y?
>functional_3/functional_1/category_encoding_7/bincount/GreaterGreaterDfunctional_3/functional_1/category_encoding_7/bincount/Prod:output:0Ifunctional_3/functional_1/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_7/bincount/Greater?
;functional_3/functional_1/category_encoding_7/bincount/CastCastBfunctional_3/functional_1/category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2=
;functional_3/functional_1/category_encoding_7/bincount/Cast?
>functional_3/functional_1/category_encoding_7/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2@
>functional_3/functional_1/category_encoding_7/bincount/Const_1?
:functional_3/functional_1/category_encoding_7/bincount/MaxMax[functional_3/functional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0Gfunctional_3/functional_1/category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_7/bincount/Max?
<functional_3/functional_1/category_encoding_7/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2>
<functional_3/functional_1/category_encoding_7/bincount/add/y?
:functional_3/functional_1/category_encoding_7/bincount/addAddV2Cfunctional_3/functional_1/category_encoding_7/bincount/Max:output:0Efunctional_3/functional_1/category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_7/bincount/add?
:functional_3/functional_1/category_encoding_7/bincount/mulMul?functional_3/functional_1/category_encoding_7/bincount/Cast:y:0>functional_3/functional_1/category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2<
:functional_3/functional_1/category_encoding_7/bincount/mul?
@functional_3/functional_1/category_encoding_7/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2B
@functional_3/functional_1/category_encoding_7/bincount/minlength?
>functional_3/functional_1/category_encoding_7/bincount/MaximumMaximumIfunctional_3/functional_1/category_encoding_7/bincount/minlength:output:0>functional_3/functional_1/category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 2@
>functional_3/functional_1/category_encoding_7/bincount/Maximum?
>functional_3/functional_1/category_encoding_7/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2@
>functional_3/functional_1/category_encoding_7/bincount/Const_2?
Dfunctional_3/functional_1/category_encoding_7/bincount/DenseBincountDenseBincount[functional_3/functional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0Bfunctional_3/functional_1/category_encoding_7/bincount/Maximum:z:0Gfunctional_3/functional_1/category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2F
Dfunctional_3/functional_1/category_encoding_7/bincount/DenseBincount?
3functional_3/functional_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :25
3functional_3/functional_1/concatenate_1/concat/axis?
.functional_3/functional_1/concatenate_1/concatConcatV23functional_3/functional_1/normalization/truediv:z:0Kfunctional_3/functional_1/category_encoding/bincount/DenseBincount:output:0Mfunctional_3/functional_1/category_encoding_1/bincount/DenseBincount:output:0Mfunctional_3/functional_1/category_encoding_2/bincount/DenseBincount:output:0Mfunctional_3/functional_1/category_encoding_3/bincount/DenseBincount:output:0Mfunctional_3/functional_1/category_encoding_4/bincount/DenseBincount:output:0Mfunctional_3/functional_1/category_encoding_5/bincount/DenseBincount:output:0Mfunctional_3/functional_1/category_encoding_6/bincount/DenseBincount:output:0Mfunctional_3/functional_1/category_encoding_7/bincount/DenseBincount:output:0<functional_3/functional_1/concatenate_1/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????|20
.functional_3/functional_1/concatenate_1/concat?
3functional_3/sequential/dense/MatMul/ReadVariableOpReadVariableOp<functional_3_sequential_dense_matmul_readvariableop_resource*
_output_shapes

:|@*
dtype025
3functional_3/sequential/dense/MatMul/ReadVariableOp?
$functional_3/sequential/dense/MatMulMatMul7functional_3/functional_1/concatenate_1/concat:output:0;functional_3/sequential/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2&
$functional_3/sequential/dense/MatMul?
4functional_3/sequential/dense/BiasAdd/ReadVariableOpReadVariableOp=functional_3_sequential_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype026
4functional_3/sequential/dense/BiasAdd/ReadVariableOp?
%functional_3/sequential/dense/BiasAddBiasAdd.functional_3/sequential/dense/MatMul:product:0<functional_3/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2'
%functional_3/sequential/dense/BiasAdd?
9functional_3/sequential/predictions/MatMul/ReadVariableOpReadVariableOpBfunctional_3_sequential_predictions_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02;
9functional_3/sequential/predictions/MatMul/ReadVariableOp?
*functional_3/sequential/predictions/MatMulMatMul.functional_3/sequential/dense/BiasAdd:output:0Afunctional_3/sequential/predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2,
*functional_3/sequential/predictions/MatMul?
:functional_3/sequential/predictions/BiasAdd/ReadVariableOpReadVariableOpCfunctional_3_sequential_predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02<
:functional_3/sequential/predictions/BiasAdd/ReadVariableOp?
+functional_3/sequential/predictions/BiasAddBiasAdd4functional_3/sequential/predictions/MatMul:product:0Bfunctional_3/sequential/predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2-
+functional_3/sequential/predictions/BiasAdd?
+functional_3/sequential/predictions/SoftmaxSoftmax4functional_3/sequential/predictions/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2-
+functional_3/sequential/predictions/Softmax?
IdentityIdentity5functional_3/sequential/predictions/Softmax:softmax:0Q^functional_3/functional_1/string_lookup/None_lookup_table_find/LookupTableFindV2S^functional_3/functional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2S^functional_3/functional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2S^functional_3/functional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2S^functional_3/functional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2S^functional_3/functional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2S^functional_3/functional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2S^functional_3/functional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::2?
Pfunctional_3/functional_1/string_lookup/None_lookup_table_find/LookupTableFindV2Pfunctional_3/functional_1/string_lookup/None_lookup_table_find/LookupTableFindV22?
Rfunctional_3/functional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV2Rfunctional_3/functional_1/string_lookup_1/None_lookup_table_find/LookupTableFindV22?
Rfunctional_3/functional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV2Rfunctional_3/functional_1/string_lookup_2/None_lookup_table_find/LookupTableFindV22?
Rfunctional_3/functional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV2Rfunctional_3/functional_1/string_lookup_3/None_lookup_table_find/LookupTableFindV22?
Rfunctional_3/functional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV2Rfunctional_3/functional_1/string_lookup_4/None_lookup_table_find/LookupTableFindV22?
Rfunctional_3/functional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV2Rfunctional_3/functional_1/string_lookup_5/None_lookup_table_find/LookupTableFindV22?
Rfunctional_3/functional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV2Rfunctional_3/functional_1/string_lookup_6/None_lookup_table_find/LookupTableFindV22?
Rfunctional_3/functional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2Rfunctional_3/functional_1/string_lookup_7/None_lookup_table_find/LookupTableFindV2:L H
'
_output_shapes
:?????????

_user_specified_nameage:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-gain:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-loss:RN
'
_output_shapes
:?????????
#
_user_specified_name	education:VR
'
_output_shapes
:?????????
'
_user_specified_nameeducation-num:OK
'
_output_shapes
:?????????
 
_user_specified_namefnlwgt:WS
'
_output_shapes
:?????????
(
_user_specified_namehours-per-week:WS
'
_output_shapes
:?????????
(
_user_specified_namemarital-status:WS
'
_output_shapes
:?????????
(
_user_specified_namenative-country:S	O
'
_output_shapes
:?????????
$
_user_specified_name
occupation:M
I
'
_output_shapes
:?????????

_user_specified_namerace:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelationship:LH
'
_output_shapes
:?????????

_user_specified_namesex:RN
'
_output_shapes
:?????????
#
_user_specified_name	workclass
?
K
__inference__creator_15067
identity??string_lookup_2_index_table?
string_lookup_2_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_118*
value_dtype0	2
string_lookup_2_index_table?
IdentityIdentity*string_lookup_2_index_table:table_handle:0^string_lookup_2_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 2:
string_lookup_2_index_tablestring_lookup_2_index_table
?
,
__inference__destroyer_15122
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
__inference_save_fn_15306
checkpoint_key[
Wstring_lookup_5_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Jstring_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2?
Jstring_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_5_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*'
_output_shapes
:?????????:2L
Jstring_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0K^string_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0K^string_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityQstring_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2*
T0*#
_output_shapes
:?????????2

Identity_2?

Identity_3Identity	add_1:z:0K^string_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0K^string_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentitySstring_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Jstring_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_5_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
K
__inference__creator_15112
identity??string_lookup_5_index_table?
string_lookup_5_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_214*
value_dtype0	2
string_lookup_5_index_table?
IdentityIdentity*string_lookup_5_index_table:table_handle:0^string_lookup_5_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 2:
string_lookup_5_index_tablestring_lookup_5_index_table
?
,
__inference__destroyer_15152
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?[
?
__inference__traced_save_15566
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop<
8savev2_sequential_predictions_kernel_read_readvariableop:
6savev2_sequential_predictions_bias_read_readvariableopS
Osavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2U
Qsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1	U
Qsavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2W
Ssavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_1	U
Qsavev2_string_lookup_2_index_table_lookup_table_export_values_lookuptableexportv2W
Ssavev2_string_lookup_2_index_table_lookup_table_export_values_lookuptableexportv2_1	U
Qsavev2_string_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2W
Ssavev2_string_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2_1	U
Qsavev2_string_lookup_4_index_table_lookup_table_export_values_lookuptableexportv2W
Ssavev2_string_lookup_4_index_table_lookup_table_export_values_lookuptableexportv2_1	U
Qsavev2_string_lookup_5_index_table_lookup_table_export_values_lookuptableexportv2W
Ssavev2_string_lookup_5_index_table_lookup_table_export_values_lookuptableexportv2_1	U
Qsavev2_string_lookup_6_index_table_lookup_table_export_values_lookuptableexportv2W
Ssavev2_string_lookup_6_index_table_lookup_table_export_values_lookuptableexportv2_1	U
Qsavev2_string_lookup_7_index_table_lookup_table_export_values_lookuptableexportv2W
Ssavev2_string_lookup_7_index_table_lookup_table_export_values_lookuptableexportv2_1	$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop=
9savev2_adam_sequential_dense_kernel_m_read_readvariableop;
7savev2_adam_sequential_dense_bias_m_read_readvariableopC
?savev2_adam_sequential_predictions_kernel_m_read_readvariableopA
=savev2_adam_sequential_predictions_bias_m_read_readvariableop=
9savev2_adam_sequential_dense_kernel_v_read_readvariableop;
7savev2_adam_sequential_dense_bias_v_read_readvariableopC
?savev2_adam_sequential_predictions_kernel_v_read_readvariableopA
=savev2_adam_sequential_predictions_bias_v_read_readvariableop
savev2_const_8

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
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_ac1fa2e70e9345d385b8d002ff30e07c/part2	
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*?
value?B?'B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEBGlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-0/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-1/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-1/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-2/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-2/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-3/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-3/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-4/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-4/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-5/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-5/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-6/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-6/_table/.ATTRIBUTES/table-valuesBGlayer_with_weights-0/layer_with_weights-7/_table/.ATTRIBUTES/table-keysBIlayer_with_weights-0/layer_with_weights-7/_table/.ATTRIBUTES/table-valuesB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopsavev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop2savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop8savev2_sequential_predictions_kernel_read_readvariableop6savev2_sequential_predictions_bias_read_readvariableopOsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2Qsavev2_string_lookup_index_table_lookup_table_export_values_lookuptableexportv2_1Qsavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_1Qsavev2_string_lookup_2_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_2_index_table_lookup_table_export_values_lookuptableexportv2_1Qsavev2_string_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2_1Qsavev2_string_lookup_4_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_4_index_table_lookup_table_export_values_lookuptableexportv2_1Qsavev2_string_lookup_5_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_5_index_table_lookup_table_export_values_lookuptableexportv2_1Qsavev2_string_lookup_6_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_6_index_table_lookup_table_export_values_lookuptableexportv2_1Qsavev2_string_lookup_7_index_table_lookup_table_export_values_lookuptableexportv2Ssavev2_string_lookup_7_index_table_lookup_table_export_values_lookuptableexportv2_1 savev2_total_read_readvariableop"savev2_count_1_read_readvariableop9savev2_adam_sequential_dense_kernel_m_read_readvariableop7savev2_adam_sequential_dense_bias_m_read_readvariableop?savev2_adam_sequential_predictions_kernel_m_read_readvariableop=savev2_adam_sequential_predictions_bias_m_read_readvariableop9savev2_adam_sequential_dense_kernel_v_read_readvariableop7savev2_adam_sequential_dense_bias_v_read_readvariableop?savev2_adam_sequential_predictions_kernel_v_read_readvariableop=savev2_adam_sequential_predictions_bias_v_read_readvariableopsavev2_const_8"/device:CPU:0*
_output_shapes
 *5
dtypes+
)2'										2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : ::: :|@:@:@::?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????: : :|@:@:@::|@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :$	 

_output_shapes

:|@: 


_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:)%
#
_output_shapes
:?????????:

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:|@:  

_output_shapes
:@:$! 

_output_shapes

:@: "

_output_shapes
::$# 

_output_shapes

:|@: $

_output_shapes
:@:$% 

_output_shapes

:@: &

_output_shapes
::'

_output_shapes
: 
?
.
__inference__initializer_15042
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
K
__inference__creator_15142
identity??string_lookup_7_index_table?
string_lookup_7_index_tableMutableHashTableV2*
_output_shapes
: *
	key_dtype0*
shared_name	table_278*
value_dtype0	2
string_lookup_7_index_table?
IdentityIdentity*string_lookup_7_index_table:table_handle:0^string_lookup_7_index_table*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 2:
string_lookup_7_index_tablestring_lookup_7_index_table
?
?
H__inference_concatenate_1_layer_call_and_return_conditional_losses_12348

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
inputs_8
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7inputs_8concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????|2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????	:?????????:?????????:?????????:?????????:?????????,:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????	
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????
 
_user_specified_nameinputs:OK
'
_output_shapes
:?????????,
 
_user_specified_nameinputs
?	
?
__inference_restore_fn_15179
restored_tensors_0
restored_tensors_1	L
Hstring_lookup_index_table_table_restore_lookuptableimportv2_table_handle
identity??;string_lookup_index_table_table_restore/LookupTableImportV2?
;string_lookup_index_table_table_restore/LookupTableImportV2LookupTableImportV2Hstring_lookup_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2=
;string_lookup_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0<^string_lookup_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::2z
;string_lookup_index_table_table_restore/LookupTableImportV2;string_lookup_index_table_table_restore/LookupTableImportV2:W S
#
_output_shapes
:?????????
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?!
?
G__inference_functional_3_layer_call_and_return_conditional_losses_13384
age
capital_gain
capital_loss
	education
education_num

fnlwgt
hours_per_week
marital_status
native_country

occupation
race
relationship
sex
	workclass
functional_1_13337
functional_1_13339	
functional_1_13341
functional_1_13343	
functional_1_13345
functional_1_13347	
functional_1_13349
functional_1_13351	
functional_1_13353
functional_1_13355	
functional_1_13357
functional_1_13359	
functional_1_13361
functional_1_13363	
functional_1_13365
functional_1_13367	
functional_1_13369
functional_1_13371
sequential_13374
sequential_13376
sequential_13378
sequential_13380
identity??$functional_1/StatefulPartitionedCall?"sequential/StatefulPartitionedCall?
$functional_1/StatefulPartitionedCallStatefulPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclassfunctional_1_13337functional_1_13339functional_1_13341functional_1_13343functional_1_13345functional_1_13347functional_1_13349functional_1_13351functional_1_13353functional_1_13355functional_1_13357functional_1_13359functional_1_13361functional_1_13363functional_1_13365functional_1_13367functional_1_13369functional_1_13371*+
Tin$
"2 								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????|*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_129572&
$functional_1/StatefulPartitionedCall?
"sequential/StatefulPartitionedCallStatefulPartitionedCall-functional_1/StatefulPartitionedCall:output:0sequential_13374sequential_13376sequential_13378sequential_13380*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_131122$
"sequential/StatefulPartitionedCall?
IdentityIdentity+sequential/StatefulPartitionedCall:output:0%^functional_1/StatefulPartitionedCall#^sequential/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::2L
$functional_1/StatefulPartitionedCall$functional_1/StatefulPartitionedCall2H
"sequential/StatefulPartitionedCall"sequential/StatefulPartitionedCall:L H
'
_output_shapes
:?????????

_user_specified_nameage:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-gain:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-loss:RN
'
_output_shapes
:?????????
#
_user_specified_name	education:VR
'
_output_shapes
:?????????
'
_user_specified_nameeducation-num:OK
'
_output_shapes
:?????????
 
_user_specified_namefnlwgt:WS
'
_output_shapes
:?????????
(
_user_specified_namehours-per-week:WS
'
_output_shapes
:?????????
(
_user_specified_namemarital-status:WS
'
_output_shapes
:?????????
(
_user_specified_namenative-country:S	O
'
_output_shapes
:?????????
$
_user_specified_name
occupation:M
I
'
_output_shapes
:?????????

_user_specified_namerace:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelationship:LH
'
_output_shapes
:?????????

_user_specified_namesex:RN
'
_output_shapes
:?????????
#
_user_specified_name	workclass
?
?
__inference_save_fn_15198
checkpoint_key[
Wstring_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2?
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_1_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*'
_output_shapes
:?????????:2L
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityQstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*#
_output_shapes
:?????????2

Identity_2?

Identity_3Identity	add_1:z:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentitySstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_1_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
??
?
G__inference_functional_1_layer_call_and_return_conditional_losses_14601

inputs_age
inputs_capital_gain
inputs_capital_loss
inputs_education
inputs_education_num
inputs_fnlwgt
inputs_hours_per_week
inputs_marital_status
inputs_native_country
inputs_occupation
inputs_race
inputs_relationship

inputs_sex
inputs_workclassI
Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value	I
Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleJ
Fstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value	G
Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleH
Dstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value	1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
identity??6string_lookup/None_lookup_table_find/LookupTableFindV2?8string_lookup_1/None_lookup_table_find/LookupTableFindV2?8string_lookup_2/None_lookup_table_find/LookupTableFindV2?8string_lookup_3/None_lookup_table_find/LookupTableFindV2?8string_lookup_4/None_lookup_table_find/LookupTableFindV2?8string_lookup_5/None_lookup_table_find/LookupTableFindV2?8string_lookup_6/None_lookup_table_find/LookupTableFindV2?8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_7/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_7_none_lookup_table_find_lookuptablefindv2_table_handleinputs_native_countryFstring_lookup_7_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_7/None_lookup_table_find/LookupTableFindV2?
8string_lookup_6/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_6_none_lookup_table_find_lookuptablefindv2_table_handle
inputs_sexFstring_lookup_6_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_6/None_lookup_table_find/LookupTableFindV2?
8string_lookup_5/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_5_none_lookup_table_find_lookuptablefindv2_table_handleinputs_raceFstring_lookup_5_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_5/None_lookup_table_find/LookupTableFindV2?
8string_lookup_4/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_4_none_lookup_table_find_lookuptablefindv2_table_handleinputs_relationshipFstring_lookup_4_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_4/None_lookup_table_find/LookupTableFindV2?
8string_lookup_3/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_3_none_lookup_table_find_lookuptablefindv2_table_handleinputs_occupationFstring_lookup_3_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_3/None_lookup_table_find/LookupTableFindV2?
8string_lookup_2/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_2_none_lookup_table_find_lookuptablefindv2_table_handleinputs_marital_statusFstring_lookup_2_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_2/None_lookup_table_find/LookupTableFindV2?
8string_lookup_1/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Estring_lookup_1_none_lookup_table_find_lookuptablefindv2_table_handleinputs_educationFstring_lookup_1_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????2:
8string_lookup_1/None_lookup_table_find/LookupTableFindV2?
6string_lookup/None_lookup_table_find/LookupTableFindV2LookupTableFindV2Cstring_lookup_none_lookup_table_find_lookuptablefindv2_table_handleinputs_workclassDstring_lookup_none_lookup_table_find_lookuptablefindv2_default_value",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*'
_output_shapes
:?????????28
6string_lookup/None_lookup_table_find/LookupTableFindV2t
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2
inputs_ageinputs_fnlwgtinputs_education_numinputs_capital_gaininputs_capital_lossinputs_hours_per_week concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:?????????2
concatenate/concat?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1?
normalization/subSubconcatenate/concat:output:0normalization/Reshape:output:0*
T0*'
_output_shapes
:?????????2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrt?
normalization/truedivRealDivnormalization/sub:z:0normalization/Sqrt:y:0*
T0*'
_output_shapes
:?????????2
normalization/truediv?
 category_encoding/bincount/ShapeShape?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2"
 category_encoding/bincount/Shape?
 category_encoding/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2"
 category_encoding/bincount/Const?
category_encoding/bincount/ProdProd)category_encoding/bincount/Shape:output:0)category_encoding/bincount/Const:output:0*
T0*
_output_shapes
: 2!
category_encoding/bincount/Prod?
$category_encoding/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2&
$category_encoding/bincount/Greater/y?
"category_encoding/bincount/GreaterGreater(category_encoding/bincount/Prod:output:0-category_encoding/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2$
"category_encoding/bincount/Greater?
category_encoding/bincount/CastCast&category_encoding/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2!
category_encoding/bincount/Cast?
"category_encoding/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2$
"category_encoding/bincount/Const_1?
category_encoding/bincount/MaxMax?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0+category_encoding/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/Max?
 category_encoding/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2"
 category_encoding/bincount/add/y?
category_encoding/bincount/addAddV2'category_encoding/bincount/Max:output:0)category_encoding/bincount/add/y:output:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/add?
category_encoding/bincount/mulMul#category_encoding/bincount/Cast:y:0"category_encoding/bincount/add:z:0*
T0	*
_output_shapes
: 2 
category_encoding/bincount/mul?
$category_encoding/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2&
$category_encoding/bincount/minlength?
"category_encoding/bincount/MaximumMaximum-category_encoding/bincount/minlength:output:0"category_encoding/bincount/mul:z:0*
T0	*
_output_shapes
: 2$
"category_encoding/bincount/Maximum?
"category_encoding/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2$
"category_encoding/bincount/Const_2?
(category_encoding/bincount/DenseBincountDenseBincount?string_lookup/None_lookup_table_find/LookupTableFindV2:values:0&category_encoding/bincount/Maximum:z:0+category_encoding/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2*
(category_encoding/bincount/DenseBincount?
"category_encoding_1/bincount/ShapeShapeAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_1/bincount/Shape?
"category_encoding_1/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_1/bincount/Const?
!category_encoding_1/bincount/ProdProd+category_encoding_1/bincount/Shape:output:0+category_encoding_1/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_1/bincount/Prod?
&category_encoding_1/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_1/bincount/Greater/y?
$category_encoding_1/bincount/GreaterGreater*category_encoding_1/bincount/Prod:output:0/category_encoding_1/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_1/bincount/Greater?
!category_encoding_1/bincount/CastCast(category_encoding_1/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_1/bincount/Cast?
$category_encoding_1/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_1/bincount/Const_1?
 category_encoding_1/bincount/MaxMaxAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_1/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/Max?
"category_encoding_1/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_1/bincount/add/y?
 category_encoding_1/bincount/addAddV2)category_encoding_1/bincount/Max:output:0+category_encoding_1/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/add?
 category_encoding_1/bincount/mulMul%category_encoding_1/bincount/Cast:y:0$category_encoding_1/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_1/bincount/mul?
&category_encoding_1/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_1/bincount/minlength?
$category_encoding_1/bincount/MaximumMaximum/category_encoding_1/bincount/minlength:output:0$category_encoding_1/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_1/bincount/Maximum?
$category_encoding_1/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_1/bincount/Const_2?
*category_encoding_1/bincount/DenseBincountDenseBincountAstring_lookup_1/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_1/bincount/Maximum:z:0-category_encoding_1/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_1/bincount/DenseBincount?
"category_encoding_2/bincount/ShapeShapeAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_2/bincount/Shape?
"category_encoding_2/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_2/bincount/Const?
!category_encoding_2/bincount/ProdProd+category_encoding_2/bincount/Shape:output:0+category_encoding_2/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_2/bincount/Prod?
&category_encoding_2/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_2/bincount/Greater/y?
$category_encoding_2/bincount/GreaterGreater*category_encoding_2/bincount/Prod:output:0/category_encoding_2/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_2/bincount/Greater?
!category_encoding_2/bincount/CastCast(category_encoding_2/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_2/bincount/Cast?
$category_encoding_2/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_2/bincount/Const_1?
 category_encoding_2/bincount/MaxMaxAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_2/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/Max?
"category_encoding_2/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_2/bincount/add/y?
 category_encoding_2/bincount/addAddV2)category_encoding_2/bincount/Max:output:0+category_encoding_2/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/add?
 category_encoding_2/bincount/mulMul%category_encoding_2/bincount/Cast:y:0$category_encoding_2/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_2/bincount/mul?
&category_encoding_2/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R	2(
&category_encoding_2/bincount/minlength?
$category_encoding_2/bincount/MaximumMaximum/category_encoding_2/bincount/minlength:output:0$category_encoding_2/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_2/bincount/Maximum?
$category_encoding_2/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_2/bincount/Const_2?
*category_encoding_2/bincount/DenseBincountDenseBincountAstring_lookup_2/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_2/bincount/Maximum:z:0-category_encoding_2/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????	*
binary_output(2,
*category_encoding_2/bincount/DenseBincount?
"category_encoding_3/bincount/ShapeShapeAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_3/bincount/Shape?
"category_encoding_3/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_3/bincount/Const?
!category_encoding_3/bincount/ProdProd+category_encoding_3/bincount/Shape:output:0+category_encoding_3/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_3/bincount/Prod?
&category_encoding_3/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_3/bincount/Greater/y?
$category_encoding_3/bincount/GreaterGreater*category_encoding_3/bincount/Prod:output:0/category_encoding_3/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_3/bincount/Greater?
!category_encoding_3/bincount/CastCast(category_encoding_3/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_3/bincount/Cast?
$category_encoding_3/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_3/bincount/Const_1?
 category_encoding_3/bincount/MaxMaxAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_3/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/Max?
"category_encoding_3/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_3/bincount/add/y?
 category_encoding_3/bincount/addAddV2)category_encoding_3/bincount/Max:output:0+category_encoding_3/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/add?
 category_encoding_3/bincount/mulMul%category_encoding_3/bincount/Cast:y:0$category_encoding_3/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_3/bincount/mul?
&category_encoding_3/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_3/bincount/minlength?
$category_encoding_3/bincount/MaximumMaximum/category_encoding_3/bincount/minlength:output:0$category_encoding_3/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_3/bincount/Maximum?
$category_encoding_3/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_3/bincount/Const_2?
*category_encoding_3/bincount/DenseBincountDenseBincountAstring_lookup_3/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_3/bincount/Maximum:z:0-category_encoding_3/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_3/bincount/DenseBincount?
"category_encoding_4/bincount/ShapeShapeAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_4/bincount/Shape?
"category_encoding_4/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_4/bincount/Const?
!category_encoding_4/bincount/ProdProd+category_encoding_4/bincount/Shape:output:0+category_encoding_4/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_4/bincount/Prod?
&category_encoding_4/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_4/bincount/Greater/y?
$category_encoding_4/bincount/GreaterGreater*category_encoding_4/bincount/Prod:output:0/category_encoding_4/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_4/bincount/Greater?
!category_encoding_4/bincount/CastCast(category_encoding_4/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_4/bincount/Cast?
$category_encoding_4/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_4/bincount/Const_1?
 category_encoding_4/bincount/MaxMaxAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_4/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/Max?
"category_encoding_4/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_4/bincount/add/y?
 category_encoding_4/bincount/addAddV2)category_encoding_4/bincount/Max:output:0+category_encoding_4/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/add?
 category_encoding_4/bincount/mulMul%category_encoding_4/bincount/Cast:y:0$category_encoding_4/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_4/bincount/mul?
&category_encoding_4/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_4/bincount/minlength?
$category_encoding_4/bincount/MaximumMaximum/category_encoding_4/bincount/minlength:output:0$category_encoding_4/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_4/bincount/Maximum?
$category_encoding_4/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_4/bincount/Const_2?
*category_encoding_4/bincount/DenseBincountDenseBincountAstring_lookup_4/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_4/bincount/Maximum:z:0-category_encoding_4/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_4/bincount/DenseBincount?
"category_encoding_5/bincount/ShapeShapeAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_5/bincount/Shape?
"category_encoding_5/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_5/bincount/Const?
!category_encoding_5/bincount/ProdProd+category_encoding_5/bincount/Shape:output:0+category_encoding_5/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_5/bincount/Prod?
&category_encoding_5/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_5/bincount/Greater/y?
$category_encoding_5/bincount/GreaterGreater*category_encoding_5/bincount/Prod:output:0/category_encoding_5/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_5/bincount/Greater?
!category_encoding_5/bincount/CastCast(category_encoding_5/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_5/bincount/Cast?
$category_encoding_5/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_5/bincount/Const_1?
 category_encoding_5/bincount/MaxMaxAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_5/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/Max?
"category_encoding_5/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_5/bincount/add/y?
 category_encoding_5/bincount/addAddV2)category_encoding_5/bincount/Max:output:0+category_encoding_5/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/add?
 category_encoding_5/bincount/mulMul%category_encoding_5/bincount/Cast:y:0$category_encoding_5/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_5/bincount/mul?
&category_encoding_5/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_5/bincount/minlength?
$category_encoding_5/bincount/MaximumMaximum/category_encoding_5/bincount/minlength:output:0$category_encoding_5/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_5/bincount/Maximum?
$category_encoding_5/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_5/bincount/Const_2?
*category_encoding_5/bincount/DenseBincountDenseBincountAstring_lookup_5/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_5/bincount/Maximum:z:0-category_encoding_5/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_5/bincount/DenseBincount?
"category_encoding_6/bincount/ShapeShapeAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_6/bincount/Shape?
"category_encoding_6/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_6/bincount/Const?
!category_encoding_6/bincount/ProdProd+category_encoding_6/bincount/Shape:output:0+category_encoding_6/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_6/bincount/Prod?
&category_encoding_6/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_6/bincount/Greater/y?
$category_encoding_6/bincount/GreaterGreater*category_encoding_6/bincount/Prod:output:0/category_encoding_6/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_6/bincount/Greater?
!category_encoding_6/bincount/CastCast(category_encoding_6/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_6/bincount/Cast?
$category_encoding_6/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_6/bincount/Const_1?
 category_encoding_6/bincount/MaxMaxAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_6/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/Max?
"category_encoding_6/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_6/bincount/add/y?
 category_encoding_6/bincount/addAddV2)category_encoding_6/bincount/Max:output:0+category_encoding_6/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/add?
 category_encoding_6/bincount/mulMul%category_encoding_6/bincount/Cast:y:0$category_encoding_6/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_6/bincount/mul?
&category_encoding_6/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R2(
&category_encoding_6/bincount/minlength?
$category_encoding_6/bincount/MaximumMaximum/category_encoding_6/bincount/minlength:output:0$category_encoding_6/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_6/bincount/Maximum?
$category_encoding_6/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_6/bincount/Const_2?
*category_encoding_6/bincount/DenseBincountDenseBincountAstring_lookup_6/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_6/bincount/Maximum:z:0-category_encoding_6/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????*
binary_output(2,
*category_encoding_6/bincount/DenseBincount?
"category_encoding_7/bincount/ShapeShapeAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0*
T0	*
_output_shapes
:2$
"category_encoding_7/bincount/Shape?
"category_encoding_7/bincount/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2$
"category_encoding_7/bincount/Const?
!category_encoding_7/bincount/ProdProd+category_encoding_7/bincount/Shape:output:0+category_encoding_7/bincount/Const:output:0*
T0*
_output_shapes
: 2#
!category_encoding_7/bincount/Prod?
&category_encoding_7/bincount/Greater/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&category_encoding_7/bincount/Greater/y?
$category_encoding_7/bincount/GreaterGreater*category_encoding_7/bincount/Prod:output:0/category_encoding_7/bincount/Greater/y:output:0*
T0*
_output_shapes
: 2&
$category_encoding_7/bincount/Greater?
!category_encoding_7/bincount/CastCast(category_encoding_7/bincount/Greater:z:0*

DstT0	*

SrcT0
*
_output_shapes
: 2#
!category_encoding_7/bincount/Cast?
$category_encoding_7/bincount/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2&
$category_encoding_7/bincount/Const_1?
 category_encoding_7/bincount/MaxMaxAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0-category_encoding_7/bincount/Const_1:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/Max?
"category_encoding_7/bincount/add/yConst*
_output_shapes
: *
dtype0	*
value	B	 R2$
"category_encoding_7/bincount/add/y?
 category_encoding_7/bincount/addAddV2)category_encoding_7/bincount/Max:output:0+category_encoding_7/bincount/add/y:output:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/add?
 category_encoding_7/bincount/mulMul%category_encoding_7/bincount/Cast:y:0$category_encoding_7/bincount/add:z:0*
T0	*
_output_shapes
: 2"
 category_encoding_7/bincount/mul?
&category_encoding_7/bincount/minlengthConst*
_output_shapes
: *
dtype0	*
value	B	 R,2(
&category_encoding_7/bincount/minlength?
$category_encoding_7/bincount/MaximumMaximum/category_encoding_7/bincount/minlength:output:0$category_encoding_7/bincount/mul:z:0*
T0	*
_output_shapes
: 2&
$category_encoding_7/bincount/Maximum?
$category_encoding_7/bincount/Const_2Const*
_output_shapes
: *
dtype0*
valueB 2&
$category_encoding_7/bincount/Const_2?
*category_encoding_7/bincount/DenseBincountDenseBincountAstring_lookup_7/None_lookup_table_find/LookupTableFindV2:values:0(category_encoding_7/bincount/Maximum:z:0-category_encoding_7/bincount/Const_2:output:0*
T0*

Tidx0	*'
_output_shapes
:?????????,*
binary_output(2,
*category_encoding_7/bincount/DenseBincountx
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2normalization/truediv:z:01category_encoding/bincount/DenseBincount:output:03category_encoding_1/bincount/DenseBincount:output:03category_encoding_2/bincount/DenseBincount:output:03category_encoding_3/bincount/DenseBincount:output:03category_encoding_4/bincount/DenseBincount:output:03category_encoding_5/bincount/DenseBincount:output:03category_encoding_6/bincount/DenseBincount:output:03category_encoding_7/bincount/DenseBincount:output:0"concatenate_1/concat/axis:output:0*
N	*
T0*'
_output_shapes
:?????????|2
concatenate_1/concat?
IdentityIdentityconcatenate_1/concat:output:07^string_lookup/None_lookup_table_find/LookupTableFindV29^string_lookup_1/None_lookup_table_find/LookupTableFindV29^string_lookup_2/None_lookup_table_find/LookupTableFindV29^string_lookup_3/None_lookup_table_find/LookupTableFindV29^string_lookup_4/None_lookup_table_find/LookupTableFindV29^string_lookup_5/None_lookup_table_find/LookupTableFindV29^string_lookup_6/None_lookup_table_find/LookupTableFindV29^string_lookup_7/None_lookup_table_find/LookupTableFindV2*
T0*'
_output_shapes
:?????????|2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::2p
6string_lookup/None_lookup_table_find/LookupTableFindV26string_lookup/None_lookup_table_find/LookupTableFindV22t
8string_lookup_1/None_lookup_table_find/LookupTableFindV28string_lookup_1/None_lookup_table_find/LookupTableFindV22t
8string_lookup_2/None_lookup_table_find/LookupTableFindV28string_lookup_2/None_lookup_table_find/LookupTableFindV22t
8string_lookup_3/None_lookup_table_find/LookupTableFindV28string_lookup_3/None_lookup_table_find/LookupTableFindV22t
8string_lookup_4/None_lookup_table_find/LookupTableFindV28string_lookup_4/None_lookup_table_find/LookupTableFindV22t
8string_lookup_5/None_lookup_table_find/LookupTableFindV28string_lookup_5/None_lookup_table_find/LookupTableFindV22t
8string_lookup_6/None_lookup_table_find/LookupTableFindV28string_lookup_6/None_lookup_table_find/LookupTableFindV22t
8string_lookup_7/None_lookup_table_find/LookupTableFindV28string_lookup_7/None_lookup_table_find/LookupTableFindV2:S O
'
_output_shapes
:?????????
$
_user_specified_name
inputs/age:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-gain:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/capital-loss:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/education:]Y
'
_output_shapes
:?????????
.
_user_specified_nameinputs/education-num:VR
'
_output_shapes
:?????????
'
_user_specified_nameinputs/fnlwgt:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/hours-per-week:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/marital-status:^Z
'
_output_shapes
:?????????
/
_user_specified_nameinputs/native-country:Z	V
'
_output_shapes
:?????????
+
_user_specified_nameinputs/occupation:T
P
'
_output_shapes
:?????????
%
_user_specified_nameinputs/race:\X
'
_output_shapes
:?????????
-
_user_specified_nameinputs/relationship:SO
'
_output_shapes
:?????????
$
_user_specified_name
inputs/sex:YU
'
_output_shapes
:?????????
*
_user_specified_nameinputs/workclass
?
?
,__inference_functional_3_layer_call_fn_13510
age
capital_gain
capital_loss
	education
education_num

fnlwgt
hours_per_week
marital_status
native_country

occupation
race
relationship
sex
	workclass
unknown
	unknown_0	
	unknown_1
	unknown_2	
	unknown_3
	unknown_4	
	unknown_5
	unknown_6	
	unknown_7
	unknown_8	
	unknown_9

unknown_10	

unknown_11

unknown_12	

unknown_13

unknown_14	

unknown_15

unknown_16

unknown_17

unknown_18

unknown_19

unknown_20
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallagecapital_gaincapital_loss	educationeducation_numfnlwgthours_per_weekmarital_statusnative_country
occupationracerelationshipsex	workclassunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_20*/
Tin(
&2$								*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

 !"#*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_3_layer_call_and_return_conditional_losses_134632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*?
_input_shapes?
?:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:?????????:: :: :: :: :: :: :: :: ::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:?????????

_user_specified_nameage:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-gain:UQ
'
_output_shapes
:?????????
&
_user_specified_namecapital-loss:RN
'
_output_shapes
:?????????
#
_user_specified_name	education:VR
'
_output_shapes
:?????????
'
_user_specified_nameeducation-num:OK
'
_output_shapes
:?????????
 
_user_specified_namefnlwgt:WS
'
_output_shapes
:?????????
(
_user_specified_namehours-per-week:WS
'
_output_shapes
:?????????
(
_user_specified_namemarital-status:WS
'
_output_shapes
:?????????
(
_user_specified_namenative-country:S	O
'
_output_shapes
:?????????
$
_user_specified_name
occupation:M
I
'
_output_shapes
:?????????

_user_specified_namerace:UQ
'
_output_shapes
:?????????
&
_user_specified_namerelationship:LH
'
_output_shapes
:?????????

_user_specified_namesex:RN
'
_output_shapes
:?????????
#
_user_specified_name	workclass
?
?
@__inference_dense_layer_call_and_return_conditional_losses_15003

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity??
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:|@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????|:::O K
'
_output_shapes
:?????????|
 
_user_specified_nameinputs
?
?
__inference_save_fn_15252
checkpoint_key[
Wstring_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2?
Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_3_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*'
_output_shapes
:?????????:2L
Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityQstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
T0*#
_output_shapes
:?????????2

Identity_2?

Identity_3Identity	add_1:z:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentitySstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_3_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
z
%__inference_dense_layer_call_fn_15012

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_130102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*.
_input_shapes
:?????????|::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????|
 
_user_specified_nameinputs
?	
?
__inference_restore_fn_15287
restored_tensors_0
restored_tensors_1	N
Jstring_lookup_4_index_table_table_restore_lookuptableimportv2_table_handle
identity??=string_lookup_4_index_table_table_restore/LookupTableImportV2?
=string_lookup_4_index_table_table_restore/LookupTableImportV2LookupTableImportV2Jstring_lookup_4_index_table_table_restore_lookuptableimportv2_table_handlerestored_tensors_0restored_tensors_1",/job:localhost/replica:0/task:0/device:CPU:0*	
Tin0*

Tout0	*
_output_shapes
 2?
=string_lookup_4_index_table_table_restore/LookupTableImportV2P
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
Const?
IdentityIdentityConst:output:0>^string_lookup_4_index_table_table_restore/LookupTableImportV2*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0**
_input_shapes
:?????????::2~
=string_lookup_4_index_table_table_restore/LookupTableImportV2=string_lookup_4_index_table_table_restore/LookupTableImportV2:W S
#
_output_shapes
:?????????
,
_user_specified_namerestored_tensors_0:LH

_output_shapes
:
,
_user_specified_namerestored_tensors_1
?
*
__inference_<lambda>_15403
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_13112

inputs
dense_13101
dense_13103
predictions_13106
predictions_13108
identity??dense/StatefulPartitionedCall?#predictions/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_13101dense_13103*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_130102
dense/StatefulPartitionedCall?
#predictions/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0predictions_13106predictions_13108*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *O
fJRH
F__inference_predictions_layer_call_and_return_conditional_losses_130372%
#predictions/StatefulPartitionedCall?
IdentityIdentity,predictions/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall$^predictions/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????|::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2J
#predictions/StatefulPartitionedCall#predictions/StatefulPartitionedCall:O K
'
_output_shapes
:?????????|
 
_user_specified_nameinputs
?
.
__inference__initializer_15147
identityP
ConstConst*
_output_shapes
: *
dtype0*
value	B :2
ConstQ
IdentityIdentityConst:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes 
?
?
__inference_save_fn_15225
checkpoint_key[
Wstring_lookup_2_index_table_lookup_table_export_values_lookuptableexportv2_table_handle
identity

identity_1

identity_2

identity_3

identity_4

identity_5	??Jstring_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2?
Jstring_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2LookupTableExportV2Wstring_lookup_2_index_table_lookup_table_export_values_lookuptableexportv2_table_handle",/job:localhost/replica:0/task:0/device:CPU:0*
Tkeys0*
Tvalues0	*'
_output_shapes
:?????????:2L
Jstring_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2T
add/yConst*
_output_shapes
: *
dtype0*
valueB B-keys2
add/yR
addAddcheckpoint_keyadd/y:output:0*
T0*
_output_shapes
: 2
addZ
add_1/yConst*
_output_shapes
: *
dtype0*
valueB B-values2	
add_1/yX
add_1Addcheckpoint_keyadd_1/y:output:0*
T0*
_output_shapes
: 2
add_1?
IdentityIdentityadd:z:0K^string_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

IdentityO
ConstConst*
_output_shapes
: *
dtype0*
valueB B 2
Const?

Identity_1IdentityConst:output:0K^string_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_1?

Identity_2IdentityQstring_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2:keys:0K^string_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2*
T0*#
_output_shapes
:?????????2

Identity_2?

Identity_3Identity	add_1:z:0K^string_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_3S
Const_1Const*
_output_shapes
: *
dtype0*
valueB B 2	
Const_1?

Identity_4IdentityConst_1:output:0K^string_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2*
T0*
_output_shapes
: 2

Identity_4?

Identity_5IdentitySstring_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2:values:0K^string_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2*
T0	*
_output_shapes
:2

Identity_5"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0*
_input_shapes
: :2?
Jstring_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2Jstring_lookup_2_index_table_lookup_table_export_values/LookupTableExportV2:F B

_output_shapes
: 
(
_user_specified_namecheckpoint_key
?
?
E__inference_sequential_layer_call_and_return_conditional_losses_14919

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource.
*predictions_matmul_readvariableop_resource/
+predictions_biasadd_readvariableop_resource
identity??
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

:|@*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense/BiasAdd?
!predictions/MatMul/ReadVariableOpReadVariableOp*predictions_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02#
!predictions/MatMul/ReadVariableOp?
predictions/MatMulMatMuldense/BiasAdd:output:0)predictions/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
predictions/MatMul?
"predictions/BiasAdd/ReadVariableOpReadVariableOp+predictions_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"predictions/BiasAdd/ReadVariableOp?
predictions/BiasAddBiasAddpredictions/MatMul:product:0*predictions/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
predictions/BiasAdd?
predictions/SoftmaxSoftmaxpredictions/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
predictions/Softmaxq
IdentityIdentitypredictions/Softmax:softmax:0*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????|:::::O K
'
_output_shapes
:?????????|
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
3
age,
serving_default_age:0?????????
E
capital-gain5
serving_default_capital-gain:0?????????
E
capital-loss5
serving_default_capital-loss:0?????????
?
	education2
serving_default_education:0?????????
G
education-num6
serving_default_education-num:0?????????
9
fnlwgt/
serving_default_fnlwgt:0?????????
I
hours-per-week7
 serving_default_hours-per-week:0?????????
I
marital-status7
 serving_default_marital-status:0?????????
I
native-country7
 serving_default_native-country:0?????????
A

occupation3
serving_default_occupation:0?????????
5
race-
serving_default_race:0?????????
E
relationship5
serving_default_relationship:0?????????
3
sex,
serving_default_sex:0?????????
?
	workclass2
serving_default_workclass:0?????????>

sequential0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer-12
layer-13
layer_with_weights-0
layer-14
layer_with_weights-1
layer-15
	optimizer
	variables
regularization_losses
trainable_variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?__call__
?_default_save_signature"??
_tf_keras_network??{"class_name": "Functional", "name": "functional_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "age"}, "name": "age", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-gain"}, "name": "capital-gain", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-loss"}, "name": "capital-loss", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "education"}, "name": "education", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "education-num"}, "name": "education-num", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "fnlwgt"}, "name": "fnlwgt", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hours-per-week"}, "name": "hours-per-week", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "marital-status"}, "name": "marital-status", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "native-country"}, "name": "native-country", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "occupation"}, "name": "occupation", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "race"}, "name": "race", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "relationship"}, "name": "relationship", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "sex"}, "name": "sex", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "workclass"}, "name": "workclass", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "age"}, "name": "age", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "fnlwgt"}, "name": "fnlwgt", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "education-num"}, "name": "education-num", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-gain"}, "name": "capital-gain", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-loss"}, "name": "capital-loss", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hours-per-week"}, "name": "hours-per-week", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "workclass"}, "name": "workclass", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "education"}, "name": "education", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "marital-status"}, "name": "marital-status", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "occupation"}, "name": "occupation", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "relationship"}, "name": "relationship", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "race"}, "name": "race", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "sex"}, "name": "sex", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "native-country"}, "name": "native-country", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["age", 0, 0, {}], ["fnlwgt", 0, 0, {}], ["education-num", 0, 0, {}], ["capital-gain", 0, 0, {}], ["capital-loss", 0, 0, {}], ["hours-per-week", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup", "inbound_nodes": [[["workclass", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_1", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_1", "inbound_nodes": [[["education", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_2", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_2", "inbound_nodes": [[["marital-status", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_3", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_3", "inbound_nodes": [[["occupation", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_4", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_4", "inbound_nodes": [[["relationship", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_5", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_5", "inbound_nodes": [[["race", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_6", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_6", "inbound_nodes": [[["sex", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_7", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_7", "inbound_nodes": [[["native-country", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding", "trainable": true, "dtype": "float32", "max_tokens": 11, "output_mode": "binary", "sparse": false}, "name": "category_encoding", "inbound_nodes": [[["string_lookup", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_1", "trainable": true, "dtype": "float32", "max_tokens": 18, "output_mode": "binary", "sparse": false}, "name": "category_encoding_1", "inbound_nodes": [[["string_lookup_1", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_2", "trainable": true, "dtype": "float32", "max_tokens": 9, "output_mode": "binary", "sparse": false}, "name": "category_encoding_2", "inbound_nodes": [[["string_lookup_2", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_3", "trainable": true, "dtype": "float32", "max_tokens": 17, "output_mode": "binary", "sparse": false}, "name": "category_encoding_3", "inbound_nodes": [[["string_lookup_3", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_4", "trainable": true, "dtype": "float32", "max_tokens": 8, "output_mode": "binary", "sparse": false}, "name": "category_encoding_4", "inbound_nodes": [[["string_lookup_4", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_5", "trainable": true, "dtype": "float32", "max_tokens": 7, "output_mode": "binary", "sparse": false}, "name": "category_encoding_5", "inbound_nodes": [[["string_lookup_5", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_6", "trainable": true, "dtype": "float32", "max_tokens": 4, "output_mode": "binary", "sparse": false}, "name": "category_encoding_6", "inbound_nodes": [[["string_lookup_6", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_7", "trainable": true, "dtype": "float32", "max_tokens": 44, "output_mode": "binary", "sparse": false}, "name": "category_encoding_7", "inbound_nodes": [[["string_lookup_7", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["normalization", 0, 0, {}], ["category_encoding", 0, 0, {}], ["category_encoding_1", 0, 0, {}], ["category_encoding_2", 0, 0, {}], ["category_encoding_3", 0, 0, {}], ["category_encoding_4", 0, 0, {}], ["category_encoding_5", 0, 0, {}], ["category_encoding_6", 0, 0, {}], ["category_encoding_7", 0, 0, {}]]]}], "input_layers": {"age": ["age", 0, 0], "workclass": ["workclass", 0, 0], "fnlwgt": ["fnlwgt", 0, 0], "education": ["education", 0, 0], "education-num": ["education-num", 0, 0], "marital-status": ["marital-status", 0, 0], "occupation": ["occupation", 0, 0], "relationship": ["relationship", 0, 0], "race": ["race", 0, 0], "sex": ["sex", 0, 0], "capital-gain": ["capital-gain", 0, 0], "capital-loss": ["capital-loss", 0, 0], "hours-per-week": ["hours-per-week", 0, 0], "native-country": ["native-country", 0, 0]}, "output_layers": [["concatenate_1", 0, 0]]}, "name": "functional_1", "inbound_nodes": [{"age": ["age", 0, 0, {}], "workclass": ["workclass", 0, 0, {}], "fnlwgt": ["fnlwgt", 0, 0, {}], "education": ["education", 0, 0, {}], "education-num": ["education-num", 0, 0, {}], "marital-status": ["marital-status", 0, 0, {}], "occupation": ["occupation", 0, 0, {}], "relationship": ["relationship", 0, 0, {}], "race": ["race", 0, 0, {}], "sex": ["sex", 0, 0, {}], "capital-gain": ["capital-gain", 0, 0, {}], "capital-loss": ["capital-loss", 0, 0, {}], "hours-per-week": ["hours-per-week", 0, 0, {}], "native-country": ["native-country", 0, 0, {}]}]}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 124]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "predictions", "trainable": true, "dtype": "float32", "units": 1, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["functional_1", 1, 0, {}]]]}], "input_layers": {"age": ["age", 0, 0], "workclass": ["workclass", 0, 0], "fnlwgt": ["fnlwgt", 0, 0], "education": ["education", 0, 0], "education-num": ["education-num", 0, 0], "marital-status": ["marital-status", 0, 0], "occupation": ["occupation", 0, 0], "relationship": ["relationship", 0, 0], "race": ["race", 0, 0], "sex": ["sex", 0, 0], "capital-gain": ["capital-gain", 0, 0], "capital-loss": ["capital-loss", 0, 0], "hours-per-week": ["hours-per-week", 0, 0], "native-country": ["native-country", 0, 0]}, "output_layers": [["sequential", 1, 0]]}, "build_input_shape": {"age": {"class_name": "TensorShape", "items": [null, 1]}, "workclass": {"class_name": "TensorShape", "items": [null, 1]}, "fnlwgt": {"class_name": "TensorShape", "items": [null, 1]}, "education": {"class_name": "TensorShape", "items": [null, 1]}, "education-num": {"class_name": "TensorShape", "items": [null, 1]}, "marital-status": {"class_name": "TensorShape", "items": [null, 1]}, "occupation": {"class_name": "TensorShape", "items": [null, 1]}, "relationship": {"class_name": "TensorShape", "items": [null, 1]}, "race": {"class_name": "TensorShape", "items": [null, 1]}, "sex": {"class_name": "TensorShape", "items": [null, 1]}, "capital-gain": {"class_name": "TensorShape", "items": [null, 1]}, "capital-loss": {"class_name": "TensorShape", "items": [null, 1]}, "hours-per-week": {"class_name": "TensorShape", "items": [null, 1]}, "native-country": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "age"}, "name": "age", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-gain"}, "name": "capital-gain", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-loss"}, "name": "capital-loss", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "education"}, "name": "education", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "education-num"}, "name": "education-num", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "fnlwgt"}, "name": "fnlwgt", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hours-per-week"}, "name": "hours-per-week", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "marital-status"}, "name": "marital-status", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "native-country"}, "name": "native-country", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "occupation"}, "name": "occupation", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "race"}, "name": "race", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "relationship"}, "name": "relationship", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "sex"}, "name": "sex", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "workclass"}, "name": "workclass", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "age"}, "name": "age", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "fnlwgt"}, "name": "fnlwgt", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "education-num"}, "name": "education-num", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-gain"}, "name": "capital-gain", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-loss"}, "name": "capital-loss", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hours-per-week"}, "name": "hours-per-week", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "workclass"}, "name": "workclass", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "education"}, "name": "education", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "marital-status"}, "name": "marital-status", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "occupation"}, "name": "occupation", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "relationship"}, "name": "relationship", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "race"}, "name": "race", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "sex"}, "name": "sex", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "native-country"}, "name": "native-country", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["age", 0, 0, {}], ["fnlwgt", 0, 0, {}], ["education-num", 0, 0, {}], ["capital-gain", 0, 0, {}], ["capital-loss", 0, 0, {}], ["hours-per-week", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup", "inbound_nodes": [[["workclass", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_1", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_1", "inbound_nodes": [[["education", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_2", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_2", "inbound_nodes": [[["marital-status", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_3", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_3", "inbound_nodes": [[["occupation", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_4", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_4", "inbound_nodes": [[["relationship", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_5", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_5", "inbound_nodes": [[["race", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_6", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_6", "inbound_nodes": [[["sex", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_7", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_7", "inbound_nodes": [[["native-country", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding", "trainable": true, "dtype": "float32", "max_tokens": 11, "output_mode": "binary", "sparse": false}, "name": "category_encoding", "inbound_nodes": [[["string_lookup", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_1", "trainable": true, "dtype": "float32", "max_tokens": 18, "output_mode": "binary", "sparse": false}, "name": "category_encoding_1", "inbound_nodes": [[["string_lookup_1", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_2", "trainable": true, "dtype": "float32", "max_tokens": 9, "output_mode": "binary", "sparse": false}, "name": "category_encoding_2", "inbound_nodes": [[["string_lookup_2", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_3", "trainable": true, "dtype": "float32", "max_tokens": 17, "output_mode": "binary", "sparse": false}, "name": "category_encoding_3", "inbound_nodes": [[["string_lookup_3", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_4", "trainable": true, "dtype": "float32", "max_tokens": 8, "output_mode": "binary", "sparse": false}, "name": "category_encoding_4", "inbound_nodes": [[["string_lookup_4", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_5", "trainable": true, "dtype": "float32", "max_tokens": 7, "output_mode": "binary", "sparse": false}, "name": "category_encoding_5", "inbound_nodes": [[["string_lookup_5", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_6", "trainable": true, "dtype": "float32", "max_tokens": 4, "output_mode": "binary", "sparse": false}, "name": "category_encoding_6", "inbound_nodes": [[["string_lookup_6", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_7", "trainable": true, "dtype": "float32", "max_tokens": 44, "output_mode": "binary", "sparse": false}, "name": "category_encoding_7", "inbound_nodes": [[["string_lookup_7", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["normalization", 0, 0, {}], ["category_encoding", 0, 0, {}], ["category_encoding_1", 0, 0, {}], ["category_encoding_2", 0, 0, {}], ["category_encoding_3", 0, 0, {}], ["category_encoding_4", 0, 0, {}], ["category_encoding_5", 0, 0, {}], ["category_encoding_6", 0, 0, {}], ["category_encoding_7", 0, 0, {}]]]}], "input_layers": {"age": ["age", 0, 0], "workclass": ["workclass", 0, 0], "fnlwgt": ["fnlwgt", 0, 0], "education": ["education", 0, 0], "education-num": ["education-num", 0, 0], "marital-status": ["marital-status", 0, 0], "occupation": ["occupation", 0, 0], "relationship": ["relationship", 0, 0], "race": ["race", 0, 0], "sex": ["sex", 0, 0], "capital-gain": ["capital-gain", 0, 0], "capital-loss": ["capital-loss", 0, 0], "hours-per-week": ["hours-per-week", 0, 0], "native-country": ["native-country", 0, 0]}, "output_layers": [["concatenate_1", 0, 0]]}, "name": "functional_1", "inbound_nodes": [{"age": ["age", 0, 0, {}], "workclass": ["workclass", 0, 0, {}], "fnlwgt": ["fnlwgt", 0, 0, {}], "education": ["education", 0, 0, {}], "education-num": ["education-num", 0, 0, {}], "marital-status": ["marital-status", 0, 0, {}], "occupation": ["occupation", 0, 0, {}], "relationship": ["relationship", 0, 0, {}], "race": ["race", 0, 0, {}], "sex": ["sex", 0, 0, {}], "capital-gain": ["capital-gain", 0, 0, {}], "capital-loss": ["capital-loss", 0, 0, {}], "hours-per-week": ["hours-per-week", 0, 0, {}], "native-country": ["native-country", 0, 0, {}]}]}, {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 124]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "predictions", "trainable": true, "dtype": "float32", "units": 1, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "name": "sequential", "inbound_nodes": [[["functional_1", 1, 0, {}]]]}], "input_layers": {"age": ["age", 0, 0], "workclass": ["workclass", 0, 0], "fnlwgt": ["fnlwgt", 0, 0], "education": ["education", 0, 0], "education-num": ["education-num", 0, 0], "marital-status": ["marital-status", 0, 0], "occupation": ["occupation", 0, 0], "relationship": ["relationship", 0, 0], "race": ["race", 0, 0], "sex": ["sex", 0, 0], "capital-gain": ["capital-gain", 0, 0], "capital-loss": ["capital-loss", 0, 0], "hours-per-week": ["hours-per-week", 0, 0], "native-country": ["native-country", 0, 0]}, "output_layers": [["sequential", 1, 0]]}}, "training_config": {"loss": "weighted_binary_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0005000000237487257, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "age", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "age"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "capital-gain", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-gain"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "capital-loss", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-loss"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "education", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "education"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "education-num", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "education-num"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "fnlwgt", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "fnlwgt"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "hours-per-week", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hours-per-week"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "marital-status", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "marital-status"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "native-country", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "native-country"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "occupation", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "occupation"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "race", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "race"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "relationship", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "relationship"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "sex", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "sex"}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "workclass", "dtype": "string", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "workclass"}}
??
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
layer-8

layer-9
layer-10
layer-11
layer-12
	layer-13
layer-14
layer_with_weights-0
layer-15
layer_with_weights-1
layer-16
layer_with_weights-2
layer-17
layer_with_weights-3
layer-18
layer_with_weights-4
layer-19
layer_with_weights-5
layer-20
layer_with_weights-6
layer-21
layer_with_weights-7
layer-22
 layer_with_weights-8
 layer-23
!layer-24
"layer-25
#layer-26
$layer-27
%layer-28
&layer-29
'layer-30
(layer-31
)layer-32
*	variables
+regularization_losses
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_networkԜ{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "age"}, "name": "age", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "fnlwgt"}, "name": "fnlwgt", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "education-num"}, "name": "education-num", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-gain"}, "name": "capital-gain", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-loss"}, "name": "capital-loss", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hours-per-week"}, "name": "hours-per-week", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "workclass"}, "name": "workclass", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "education"}, "name": "education", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "marital-status"}, "name": "marital-status", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "occupation"}, "name": "occupation", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "relationship"}, "name": "relationship", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "race"}, "name": "race", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "sex"}, "name": "sex", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "native-country"}, "name": "native-country", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["age", 0, 0, {}], ["fnlwgt", 0, 0, {}], ["education-num", 0, 0, {}], ["capital-gain", 0, 0, {}], ["capital-loss", 0, 0, {}], ["hours-per-week", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup", "inbound_nodes": [[["workclass", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_1", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_1", "inbound_nodes": [[["education", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_2", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_2", "inbound_nodes": [[["marital-status", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_3", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_3", "inbound_nodes": [[["occupation", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_4", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_4", "inbound_nodes": [[["relationship", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_5", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_5", "inbound_nodes": [[["race", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_6", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_6", "inbound_nodes": [[["sex", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_7", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_7", "inbound_nodes": [[["native-country", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding", "trainable": true, "dtype": "float32", "max_tokens": 11, "output_mode": "binary", "sparse": false}, "name": "category_encoding", "inbound_nodes": [[["string_lookup", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_1", "trainable": true, "dtype": "float32", "max_tokens": 18, "output_mode": "binary", "sparse": false}, "name": "category_encoding_1", "inbound_nodes": [[["string_lookup_1", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_2", "trainable": true, "dtype": "float32", "max_tokens": 9, "output_mode": "binary", "sparse": false}, "name": "category_encoding_2", "inbound_nodes": [[["string_lookup_2", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_3", "trainable": true, "dtype": "float32", "max_tokens": 17, "output_mode": "binary", "sparse": false}, "name": "category_encoding_3", "inbound_nodes": [[["string_lookup_3", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_4", "trainable": true, "dtype": "float32", "max_tokens": 8, "output_mode": "binary", "sparse": false}, "name": "category_encoding_4", "inbound_nodes": [[["string_lookup_4", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_5", "trainable": true, "dtype": "float32", "max_tokens": 7, "output_mode": "binary", "sparse": false}, "name": "category_encoding_5", "inbound_nodes": [[["string_lookup_5", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_6", "trainable": true, "dtype": "float32", "max_tokens": 4, "output_mode": "binary", "sparse": false}, "name": "category_encoding_6", "inbound_nodes": [[["string_lookup_6", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_7", "trainable": true, "dtype": "float32", "max_tokens": 44, "output_mode": "binary", "sparse": false}, "name": "category_encoding_7", "inbound_nodes": [[["string_lookup_7", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["normalization", 0, 0, {}], ["category_encoding", 0, 0, {}], ["category_encoding_1", 0, 0, {}], ["category_encoding_2", 0, 0, {}], ["category_encoding_3", 0, 0, {}], ["category_encoding_4", 0, 0, {}], ["category_encoding_5", 0, 0, {}], ["category_encoding_6", 0, 0, {}], ["category_encoding_7", 0, 0, {}]]]}], "input_layers": {"age": ["age", 0, 0], "workclass": ["workclass", 0, 0], "fnlwgt": ["fnlwgt", 0, 0], "education": ["education", 0, 0], "education-num": ["education-num", 0, 0], "marital-status": ["marital-status", 0, 0], "occupation": ["occupation", 0, 0], "relationship": ["relationship", 0, 0], "race": ["race", 0, 0], "sex": ["sex", 0, 0], "capital-gain": ["capital-gain", 0, 0], "capital-loss": ["capital-loss", 0, 0], "hours-per-week": ["hours-per-week", 0, 0], "native-country": ["native-country", 0, 0]}, "output_layers": [["concatenate_1", 0, 0]]}, "build_input_shape": {"age": {"class_name": "TensorShape", "items": [null, 1]}, "workclass": {"class_name": "TensorShape", "items": [null, 1]}, "fnlwgt": {"class_name": "TensorShape", "items": [null, 1]}, "education": {"class_name": "TensorShape", "items": [null, 1]}, "education-num": {"class_name": "TensorShape", "items": [null, 1]}, "marital-status": {"class_name": "TensorShape", "items": [null, 1]}, "occupation": {"class_name": "TensorShape", "items": [null, 1]}, "relationship": {"class_name": "TensorShape", "items": [null, 1]}, "race": {"class_name": "TensorShape", "items": [null, 1]}, "sex": {"class_name": "TensorShape", "items": [null, 1]}, "capital-gain": {"class_name": "TensorShape", "items": [null, 1]}, "capital-loss": {"class_name": "TensorShape", "items": [null, 1]}, "hours-per-week": {"class_name": "TensorShape", "items": [null, 1]}, "native-country": {"class_name": "TensorShape", "items": [null, 1]}}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "age"}, "name": "age", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "fnlwgt"}, "name": "fnlwgt", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "education-num"}, "name": "education-num", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-gain"}, "name": "capital-gain", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "capital-loss"}, "name": "capital-loss", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "hours-per-week"}, "name": "hours-per-week", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "workclass"}, "name": "workclass", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "education"}, "name": "education", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "marital-status"}, "name": "marital-status", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "occupation"}, "name": "occupation", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "relationship"}, "name": "relationship", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "race"}, "name": "race", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "sex"}, "name": "sex", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "string", "sparse": false, "ragged": false, "name": "native-country"}, "name": "native-country", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["age", 0, 0, {}], ["fnlwgt", 0, 0, {}], ["education-num", 0, 0, {}], ["capital-gain", 0, 0, {}], ["capital-loss", 0, 0, {}], ["hours-per-week", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup", "inbound_nodes": [[["workclass", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_1", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_1", "inbound_nodes": [[["education", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_2", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_2", "inbound_nodes": [[["marital-status", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_3", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_3", "inbound_nodes": [[["occupation", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_4", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_4", "inbound_nodes": [[["relationship", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_5", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_5", "inbound_nodes": [[["race", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_6", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_6", "inbound_nodes": [[["sex", 0, 0, {}]]]}, {"class_name": "StringLookup", "config": {"name": "string_lookup_7", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}, "name": "string_lookup_7", "inbound_nodes": [[["native-country", 0, 0, {}]]]}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "name": "normalization", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding", "trainable": true, "dtype": "float32", "max_tokens": 11, "output_mode": "binary", "sparse": false}, "name": "category_encoding", "inbound_nodes": [[["string_lookup", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_1", "trainable": true, "dtype": "float32", "max_tokens": 18, "output_mode": "binary", "sparse": false}, "name": "category_encoding_1", "inbound_nodes": [[["string_lookup_1", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_2", "trainable": true, "dtype": "float32", "max_tokens": 9, "output_mode": "binary", "sparse": false}, "name": "category_encoding_2", "inbound_nodes": [[["string_lookup_2", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_3", "trainable": true, "dtype": "float32", "max_tokens": 17, "output_mode": "binary", "sparse": false}, "name": "category_encoding_3", "inbound_nodes": [[["string_lookup_3", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_4", "trainable": true, "dtype": "float32", "max_tokens": 8, "output_mode": "binary", "sparse": false}, "name": "category_encoding_4", "inbound_nodes": [[["string_lookup_4", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_5", "trainable": true, "dtype": "float32", "max_tokens": 7, "output_mode": "binary", "sparse": false}, "name": "category_encoding_5", "inbound_nodes": [[["string_lookup_5", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_6", "trainable": true, "dtype": "float32", "max_tokens": 4, "output_mode": "binary", "sparse": false}, "name": "category_encoding_6", "inbound_nodes": [[["string_lookup_6", 0, 0, {}]]]}, {"class_name": "CategoryEncoding", "config": {"name": "category_encoding_7", "trainable": true, "dtype": "float32", "max_tokens": 44, "output_mode": "binary", "sparse": false}, "name": "category_encoding_7", "inbound_nodes": [[["string_lookup_7", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate_1", "inbound_nodes": [[["normalization", 0, 0, {}], ["category_encoding", 0, 0, {}], ["category_encoding_1", 0, 0, {}], ["category_encoding_2", 0, 0, {}], ["category_encoding_3", 0, 0, {}], ["category_encoding_4", 0, 0, {}], ["category_encoding_5", 0, 0, {}], ["category_encoding_6", 0, 0, {}], ["category_encoding_7", 0, 0, {}]]]}], "input_layers": {"age": ["age", 0, 0], "workclass": ["workclass", 0, 0], "fnlwgt": ["fnlwgt", 0, 0], "education": ["education", 0, 0], "education-num": ["education-num", 0, 0], "marital-status": ["marital-status", 0, 0], "occupation": ["occupation", 0, 0], "relationship": ["relationship", 0, 0], "race": ["race", 0, 0], "sex": ["sex", 0, 0], "capital-gain": ["capital-gain", 0, 0], "capital-loss": ["capital-loss", 0, 0], "hours-per-week": ["hours-per-week", 0, 0], "native-country": ["native-country", 0, 0]}, "output_layers": [["concatenate_1", 0, 0]]}}}
?
.layer_with_weights-0
.layer-0
/layer_with_weights-1
/layer-1
0	variables
1regularization_losses
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_sequential?{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 124]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "predictions", "trainable": true, "dtype": "float32", "units": 1, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 124}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 124]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 124]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "dense_input"}}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "predictions", "trainable": true, "dtype": "float32", "units": 1, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
?
4iter

5beta_1

6beta_2
	7decay
8learning_rate<m?=m?>m??m?<v?=v?>v??v?"
	optimizer
V
98
:9
;10
<11
=12
>13
?14"
trackable_list_wrapper
 "
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
?

@layers
	variables
Ametrics
Blayer_metrics
regularization_losses
Clayer_regularization_losses
Dnon_trainable_variables
trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?
E	variables
Fregularization_losses
Gtrainable_variables
H	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}, {"class_name": "TensorShape", "items": [null, 1]}]}
?
Istate_variables

J_table
K	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
?
Lstate_variables

M_table
N	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup_1", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup_1", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
?
Ostate_variables

P_table
Q	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup_2", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup_2", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
?
Rstate_variables

S_table
T	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup_3", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup_3", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
?
Ustate_variables

V_table
W	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup_4", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup_4", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
?
Xstate_variables

Y_table
Z	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup_5", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup_5", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
?
[state_variables

\_table
]	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup_6", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup_6", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
?
^state_variables

__table
`	keras_api"?
_tf_keras_layer?{"class_name": "StringLookup", "name": "string_lookup_7", "trainable": true, "expects_training_arg": false, "dtype": "string", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "string_lookup_7", "trainable": true, "dtype": "string", "invert": false, "max_tokens": null, "num_oov_indices": 1, "oov_token": "[UNK]", "mask_token": "", "encoding": "utf-8"}}
?
astate_variables
b_broadcast_shape
9mean
:variance
	;count
c	keras_api"?
_tf_keras_layer?{"class_name": "Normalization", "name": "normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": [512, 6]}
?
dstate_variables
e	keras_api"?
_tf_keras_layer?{"class_name": "CategoryEncoding", "name": "category_encoding", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "category_encoding", "trainable": true, "dtype": "float32", "max_tokens": 11, "output_mode": "binary", "sparse": false}}
?
fstate_variables
g	keras_api"?
_tf_keras_layer?{"class_name": "CategoryEncoding", "name": "category_encoding_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "category_encoding_1", "trainable": true, "dtype": "float32", "max_tokens": 18, "output_mode": "binary", "sparse": false}}
?
hstate_variables
i	keras_api"?
_tf_keras_layer?{"class_name": "CategoryEncoding", "name": "category_encoding_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "category_encoding_2", "trainable": true, "dtype": "float32", "max_tokens": 9, "output_mode": "binary", "sparse": false}}
?
jstate_variables
k	keras_api"?
_tf_keras_layer?{"class_name": "CategoryEncoding", "name": "category_encoding_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "category_encoding_3", "trainable": true, "dtype": "float32", "max_tokens": 17, "output_mode": "binary", "sparse": false}}
?
lstate_variables
m	keras_api"?
_tf_keras_layer?{"class_name": "CategoryEncoding", "name": "category_encoding_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "category_encoding_4", "trainable": true, "dtype": "float32", "max_tokens": 8, "output_mode": "binary", "sparse": false}}
?
nstate_variables
o	keras_api"?
_tf_keras_layer?{"class_name": "CategoryEncoding", "name": "category_encoding_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "category_encoding_5", "trainable": true, "dtype": "float32", "max_tokens": 7, "output_mode": "binary", "sparse": false}}
?
pstate_variables
q	keras_api"?
_tf_keras_layer?{"class_name": "CategoryEncoding", "name": "category_encoding_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "category_encoding_6", "trainable": true, "dtype": "float32", "max_tokens": 4, "output_mode": "binary", "sparse": false}}
?
rstate_variables
s	keras_api"?
_tf_keras_layer?{"class_name": "CategoryEncoding", "name": "category_encoding_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "config": {"name": "category_encoding_7", "trainable": true, "dtype": "float32", "max_tokens": 44, "output_mode": "binary", "sparse": false}}
?
t	variables
uregularization_losses
vtrainable_variables
w	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Concatenate", "name": "concatenate_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate_1", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 6]}, {"class_name": "TensorShape", "items": [null, 11]}, {"class_name": "TensorShape", "items": [null, 18]}, {"class_name": "TensorShape", "items": [null, 9]}, {"class_name": "TensorShape", "items": [null, 17]}, {"class_name": "TensorShape", "items": [null, 8]}, {"class_name": "TensorShape", "items": [null, 7]}, {"class_name": "TensorShape", "items": [null, 4]}, {"class_name": "TensorShape", "items": [null, 44]}]}
6
98
:9
;10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

xlayers
*	variables
ymetrics
zlayer_metrics
+regularization_losses
{layer_regularization_losses
|non_trainable_variables
,trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
}_inbound_nodes

<kernel
=bias
~	variables
regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 64, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 124}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 124]}}
?
?_inbound_nodes

>kernel
?bias
?	variables
?regularization_losses
?trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"class_name": "Dense", "name": "predictions", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "predictions", "trainable": true, "dtype": "float32", "units": 1, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
<
<0
=1
>2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
<0
=1
>2
?3"
trackable_list_wrapper
?
?layers
0	variables
?metrics
?layer_metrics
1regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
2trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
:2mean
:2variance
:	 2count
):'|@2sequential/dense/kernel
#:!@2sequential/dense/bias
/:-@2sequential/predictions/kernel
):'2sequential/predictions/bias
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
6
98
:9
;10"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
E	variables
?metrics
?layer_metrics
Fregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
Gtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
 "
trackable_dict_wrapper
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
 "
trackable_dict_wrapper
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
 "
trackable_dict_wrapper
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
 "
trackable_dict_wrapper
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
 "
trackable_dict_wrapper
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
 "
trackable_dict_wrapper
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
 "
trackable_dict_wrapper
T
?_create_resource
?_initialize
?_destroy_resourceR Z
table??
"
_generic_user_object
C
9mean
:variance
	;count"
trackable_dict_wrapper
 "
trackable_list_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_dict_wrapper
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
t	variables
?metrics
?layer_metrics
uregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
vtrainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
0
1
2
3
4
5
6
7
8

9
10
11
12
	13
14
15
16
17
18
19
20
21
22
 23
!24
"25
#26
$27
%28
&29
'30
(31
)32"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
6
98
:9
;10"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
?
?layers
~	variables
?metrics
?layer_metrics
regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
>0
?1"
trackable_list_wrapper
?
?layers
?	variables
?metrics
?layer_metrics
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
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
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
.:,|@2Adam/sequential/dense/kernel/m
(:&@2Adam/sequential/dense/bias/m
4:2@2$Adam/sequential/predictions/kernel/m
.:,2"Adam/sequential/predictions/bias/m
.:,|@2Adam/sequential/dense/kernel/v
(:&@2Adam/sequential/dense/bias/v
4:2@2$Adam/sequential/predictions/kernel/v
.:,2"Adam/sequential/predictions/bias/v
?2?
G__inference_functional_3_layer_call_and_return_conditional_losses_13384
G__inference_functional_3_layer_call_and_return_conditional_losses_14301
G__inference_functional_3_layer_call_and_return_conditional_losses_14112
G__inference_functional_3_layer_call_and_return_conditional_losses_13321?
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
?2?
,__inference_functional_3_layer_call_fn_14363
,__inference_functional_3_layer_call_fn_13635
,__inference_functional_3_layer_call_fn_13510
,__inference_functional_3_layer_call_fn_14425?
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
?2?
 __inference__wrapped_model_12138?
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
annotations? *???
???
$
age?
age?????????
6
capital-gain&?#
capital-gain?????????
6
capital-loss&?#
capital-loss?????????
0
	education#? 
	education?????????
8
education-num'?$
education-num?????????
*
fnlwgt ?
fnlwgt?????????
:
hours-per-week(?%
hours-per-week?????????
:
marital-status(?%
marital-status?????????
:
native-country(?%
native-country?????????
2

occupation$?!

occupation?????????
&
race?
race?????????
6
relationship&?#
relationship?????????
$
sex?
sex?????????
0
	workclass#? 
	workclass?????????
/B-
__inference_save_fn_15171checkpoint_key
JBH
__inference_restore_fn_15179restored_tensors_0restored_tensors_1
/B-
__inference_save_fn_15198checkpoint_key
JBH
__inference_restore_fn_15206restored_tensors_0restored_tensors_1
/B-
__inference_save_fn_15225checkpoint_key
JBH
__inference_restore_fn_15233restored_tensors_0restored_tensors_1
/B-
__inference_save_fn_15252checkpoint_key
JBH
__inference_restore_fn_15260restored_tensors_0restored_tensors_1
/B-
__inference_save_fn_15279checkpoint_key
JBH
__inference_restore_fn_15287restored_tensors_0restored_tensors_1
/B-
__inference_save_fn_15306checkpoint_key
JBH
__inference_restore_fn_15314restored_tensors_0restored_tensors_1
/B-
__inference_save_fn_15333checkpoint_key
JBH
__inference_restore_fn_15341restored_tensors_0restored_tensors_1
/B-
__inference_save_fn_15360checkpoint_key
JBH
__inference_restore_fn_15368restored_tensors_0restored_tensors_1
?2?
G__inference_functional_1_layer_call_and_return_conditional_losses_12365
G__inference_functional_1_layer_call_and_return_conditional_losses_14601
G__inference_functional_1_layer_call_and_return_conditional_losses_14777
G__inference_functional_1_layer_call_and_return_conditional_losses_12539?
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
?2?
,__inference_functional_1_layer_call_fn_14885
,__inference_functional_1_layer_call_fn_12768
,__inference_functional_1_layer_call_fn_14831
,__inference_functional_1_layer_call_fn_12996?
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
E__inference_sequential_layer_call_and_return_conditional_losses_13054
E__inference_sequential_layer_call_and_return_conditional_losses_14919
E__inference_sequential_layer_call_and_return_conditional_losses_14902
E__inference_sequential_layer_call_and_return_conditional_losses_13068?
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
?2?
*__inference_sequential_layer_call_fn_13123
*__inference_sequential_layer_call_fn_14945
*__inference_sequential_layer_call_fn_14932
*__inference_sequential_layer_call_fn_13096?
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
?B?
#__inference_signature_wrapper_13707agecapital-gaincapital-loss	educationeducation-numfnlwgthours-per-weekmarital-statusnative-country
occupationracerelationshipsex	workclass
?2?
F__inference_concatenate_layer_call_and_return_conditional_losses_14956?
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
+__inference_concatenate_layer_call_fn_14966?
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
H__inference_concatenate_1_layer_call_and_return_conditional_losses_14980?
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
-__inference_concatenate_1_layer_call_fn_14993?
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
@__inference_dense_layer_call_and_return_conditional_losses_15003?
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
%__inference_dense_layer_call_fn_15012?
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
F__inference_predictions_layer_call_and_return_conditional_losses_15023?
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
+__inference_predictions_layer_call_fn_15032?
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
__inference__creator_15037?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_15042?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_15047?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_15052?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_15057?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_15062?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_15067?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_15072?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_15077?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_15082?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_15087?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_15092?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_15097?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_15102?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_15107?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_15112?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_15117?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_15122?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_15127?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_15132?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_15137?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__creator_15142?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__initializer_15147?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference__destroyer_15152?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
	J
Const
J	
Const_1
J	
Const_2
J	
Const_3
J	
Const_4
J	
Const_5
J	
Const_6
J	
Const_76
__inference__creator_15037?

? 
? "? 6
__inference__creator_15052?

? 
? "? 6
__inference__creator_15067?

? 
? "? 6
__inference__creator_15082?

? 
? "? 6
__inference__creator_15097?

? 
? "? 6
__inference__creator_15112?

? 
? "? 6
__inference__creator_15127?

? 
? "? 6
__inference__creator_15142?

? 
? "? 8
__inference__destroyer_15047?

? 
? "? 8
__inference__destroyer_15062?

? 
? "? 8
__inference__destroyer_15077?

? 
? "? 8
__inference__destroyer_15092?

? 
? "? 8
__inference__destroyer_15107?

? 
? "? 8
__inference__destroyer_15122?

? 
? "? 8
__inference__destroyer_15137?

? 
? "? 8
__inference__destroyer_15152?

? 
? "? :
__inference__initializer_15042?

? 
? "? :
__inference__initializer_15057?

? 
? "? :
__inference__initializer_15072?

? 
? "? :
__inference__initializer_15087?

? 
? "? :
__inference__initializer_15102?

? 
? "? :
__inference__initializer_15117?

? 
? "? :
__inference__initializer_15132?

? 
? "? :
__inference__initializer_15147?

? 
? "? ?
 __inference__wrapped_model_12138?_?\?Y?V?S?P?M?J?9:<=>????
???
???
$
age?
age?????????
6
capital-gain&?#
capital-gain?????????
6
capital-loss&?#
capital-loss?????????
0
	education#? 
	education?????????
8
education-num'?$
education-num?????????
*
fnlwgt ?
fnlwgt?????????
:
hours-per-week(?%
hours-per-week?????????
:
marital-status(?%
marital-status?????????
:
native-country(?%
native-country?????????
2

occupation$?!

occupation?????????
&
race?
race?????????
6
relationship&?#
relationship?????????
$
sex?
sex?????????
0
	workclass#? 
	workclass?????????
? "7?4
2

sequential$?!

sequential??????????
H__inference_concatenate_1_layer_call_and_return_conditional_losses_14980????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????	
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????,
? "%?"
?
0?????????|
? ?
-__inference_concatenate_1_layer_call_fn_14993????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????	
"?
inputs/4?????????
"?
inputs/5?????????
"?
inputs/6?????????
"?
inputs/7?????????
"?
inputs/8?????????,
? "??????????|?
F__inference_concatenate_layer_call_and_return_conditional_losses_14956????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
? "%?"
?
0?????????
? ?
+__inference_concatenate_layer_call_fn_14966????
???
???
"?
inputs/0?????????
"?
inputs/1?????????
"?
inputs/2?????????
"?
inputs/3?????????
"?
inputs/4?????????
"?
inputs/5?????????
? "???????????
@__inference_dense_layer_call_and_return_conditional_losses_15003\<=/?,
%?"
 ?
inputs?????????|
? "%?"
?
0?????????@
? x
%__inference_dense_layer_call_fn_15012O<=/?,
%?"
 ?
inputs?????????|
? "??????????@?
G__inference_functional_1_layer_call_and_return_conditional_losses_12365?_?\?Y?V?S?P?M?J?9:???
???
???
$
age?
age?????????
6
capital-gain&?#
capital-gain?????????
6
capital-loss&?#
capital-loss?????????
0
	education#? 
	education?????????
8
education-num'?$
education-num?????????
*
fnlwgt ?
fnlwgt?????????
:
hours-per-week(?%
hours-per-week?????????
:
marital-status(?%
marital-status?????????
:
native-country(?%
native-country?????????
2

occupation$?!

occupation?????????
&
race?
race?????????
6
relationship&?#
relationship?????????
$
sex?
sex?????????
0
	workclass#? 
	workclass?????????
p

 
? "%?"
?
0?????????|
? ?
G__inference_functional_1_layer_call_and_return_conditional_losses_12539?_?\?Y?V?S?P?M?J?9:???
???
???
$
age?
age?????????
6
capital-gain&?#
capital-gain?????????
6
capital-loss&?#
capital-loss?????????
0
	education#? 
	education?????????
8
education-num'?$
education-num?????????
*
fnlwgt ?
fnlwgt?????????
:
hours-per-week(?%
hours-per-week?????????
:
marital-status(?%
marital-status?????????
:
native-country(?%
native-country?????????
2

occupation$?!

occupation?????????
&
race?
race?????????
6
relationship&?#
relationship?????????
$
sex?
sex?????????
0
	workclass#? 
	workclass?????????
p 

 
? "%?"
?
0?????????|
? ?
G__inference_functional_1_layer_call_and_return_conditional_losses_14601?_?\?Y?V?S?P?M?J?9:???
???
???
+
age$?!

inputs/age?????????
=
capital-gain-?*
inputs/capital-gain?????????
=
capital-loss-?*
inputs/capital-loss?????????
7
	education*?'
inputs/education?????????
?
education-num.?+
inputs/education-num?????????
1
fnlwgt'?$
inputs/fnlwgt?????????
A
hours-per-week/?,
inputs/hours-per-week?????????
A
marital-status/?,
inputs/marital-status?????????
A
native-country/?,
inputs/native-country?????????
9

occupation+?(
inputs/occupation?????????
-
race%?"
inputs/race?????????
=
relationship-?*
inputs/relationship?????????
+
sex$?!

inputs/sex?????????
7
	workclass*?'
inputs/workclass?????????
p

 
? "%?"
?
0?????????|
? ?
G__inference_functional_1_layer_call_and_return_conditional_losses_14777?_?\?Y?V?S?P?M?J?9:???
???
???
+
age$?!

inputs/age?????????
=
capital-gain-?*
inputs/capital-gain?????????
=
capital-loss-?*
inputs/capital-loss?????????
7
	education*?'
inputs/education?????????
?
education-num.?+
inputs/education-num?????????
1
fnlwgt'?$
inputs/fnlwgt?????????
A
hours-per-week/?,
inputs/hours-per-week?????????
A
marital-status/?,
inputs/marital-status?????????
A
native-country/?,
inputs/native-country?????????
9

occupation+?(
inputs/occupation?????????
-
race%?"
inputs/race?????????
=
relationship-?*
inputs/relationship?????????
+
sex$?!

inputs/sex?????????
7
	workclass*?'
inputs/workclass?????????
p 

 
? "%?"
?
0?????????|
? ?
,__inference_functional_1_layer_call_fn_12768?_?\?Y?V?S?P?M?J?9:???
???
???
$
age?
age?????????
6
capital-gain&?#
capital-gain?????????
6
capital-loss&?#
capital-loss?????????
0
	education#? 
	education?????????
8
education-num'?$
education-num?????????
*
fnlwgt ?
fnlwgt?????????
:
hours-per-week(?%
hours-per-week?????????
:
marital-status(?%
marital-status?????????
:
native-country(?%
native-country?????????
2

occupation$?!

occupation?????????
&
race?
race?????????
6
relationship&?#
relationship?????????
$
sex?
sex?????????
0
	workclass#? 
	workclass?????????
p

 
? "??????????|?
,__inference_functional_1_layer_call_fn_12996?_?\?Y?V?S?P?M?J?9:???
???
???
$
age?
age?????????
6
capital-gain&?#
capital-gain?????????
6
capital-loss&?#
capital-loss?????????
0
	education#? 
	education?????????
8
education-num'?$
education-num?????????
*
fnlwgt ?
fnlwgt?????????
:
hours-per-week(?%
hours-per-week?????????
:
marital-status(?%
marital-status?????????
:
native-country(?%
native-country?????????
2

occupation$?!

occupation?????????
&
race?
race?????????
6
relationship&?#
relationship?????????
$
sex?
sex?????????
0
	workclass#? 
	workclass?????????
p 

 
? "??????????|?
,__inference_functional_1_layer_call_fn_14831?_?\?Y?V?S?P?M?J?9:???
???
???
+
age$?!

inputs/age?????????
=
capital-gain-?*
inputs/capital-gain?????????
=
capital-loss-?*
inputs/capital-loss?????????
7
	education*?'
inputs/education?????????
?
education-num.?+
inputs/education-num?????????
1
fnlwgt'?$
inputs/fnlwgt?????????
A
hours-per-week/?,
inputs/hours-per-week?????????
A
marital-status/?,
inputs/marital-status?????????
A
native-country/?,
inputs/native-country?????????
9

occupation+?(
inputs/occupation?????????
-
race%?"
inputs/race?????????
=
relationship-?*
inputs/relationship?????????
+
sex$?!

inputs/sex?????????
7
	workclass*?'
inputs/workclass?????????
p

 
? "??????????|?
,__inference_functional_1_layer_call_fn_14885?_?\?Y?V?S?P?M?J?9:???
???
???
+
age$?!

inputs/age?????????
=
capital-gain-?*
inputs/capital-gain?????????
=
capital-loss-?*
inputs/capital-loss?????????
7
	education*?'
inputs/education?????????
?
education-num.?+
inputs/education-num?????????
1
fnlwgt'?$
inputs/fnlwgt?????????
A
hours-per-week/?,
inputs/hours-per-week?????????
A
marital-status/?,
inputs/marital-status?????????
A
native-country/?,
inputs/native-country?????????
9

occupation+?(
inputs/occupation?????????
-
race%?"
inputs/race?????????
=
relationship-?*
inputs/relationship?????????
+
sex$?!

inputs/sex?????????
7
	workclass*?'
inputs/workclass?????????
p 

 
? "??????????|?
G__inference_functional_3_layer_call_and_return_conditional_losses_13321?_?\?Y?V?S?P?M?J?9:<=>????
???
???
$
age?
age?????????
6
capital-gain&?#
capital-gain?????????
6
capital-loss&?#
capital-loss?????????
0
	education#? 
	education?????????
8
education-num'?$
education-num?????????
*
fnlwgt ?
fnlwgt?????????
:
hours-per-week(?%
hours-per-week?????????
:
marital-status(?%
marital-status?????????
:
native-country(?%
native-country?????????
2

occupation$?!

occupation?????????
&
race?
race?????????
6
relationship&?#
relationship?????????
$
sex?
sex?????????
0
	workclass#? 
	workclass?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_functional_3_layer_call_and_return_conditional_losses_13384?_?\?Y?V?S?P?M?J?9:<=>????
???
???
$
age?
age?????????
6
capital-gain&?#
capital-gain?????????
6
capital-loss&?#
capital-loss?????????
0
	education#? 
	education?????????
8
education-num'?$
education-num?????????
*
fnlwgt ?
fnlwgt?????????
:
hours-per-week(?%
hours-per-week?????????
:
marital-status(?%
marital-status?????????
:
native-country(?%
native-country?????????
2

occupation$?!

occupation?????????
&
race?
race?????????
6
relationship&?#
relationship?????????
$
sex?
sex?????????
0
	workclass#? 
	workclass?????????
p 

 
? "%?"
?
0?????????
? ?
G__inference_functional_3_layer_call_and_return_conditional_losses_14112?_?\?Y?V?S?P?M?J?9:<=>????
???
???
+
age$?!

inputs/age?????????
=
capital-gain-?*
inputs/capital-gain?????????
=
capital-loss-?*
inputs/capital-loss?????????
7
	education*?'
inputs/education?????????
?
education-num.?+
inputs/education-num?????????
1
fnlwgt'?$
inputs/fnlwgt?????????
A
hours-per-week/?,
inputs/hours-per-week?????????
A
marital-status/?,
inputs/marital-status?????????
A
native-country/?,
inputs/native-country?????????
9

occupation+?(
inputs/occupation?????????
-
race%?"
inputs/race?????????
=
relationship-?*
inputs/relationship?????????
+
sex$?!

inputs/sex?????????
7
	workclass*?'
inputs/workclass?????????
p

 
? "%?"
?
0?????????
? ?
G__inference_functional_3_layer_call_and_return_conditional_losses_14301?_?\?Y?V?S?P?M?J?9:<=>????
???
???
+
age$?!

inputs/age?????????
=
capital-gain-?*
inputs/capital-gain?????????
=
capital-loss-?*
inputs/capital-loss?????????
7
	education*?'
inputs/education?????????
?
education-num.?+
inputs/education-num?????????
1
fnlwgt'?$
inputs/fnlwgt?????????
A
hours-per-week/?,
inputs/hours-per-week?????????
A
marital-status/?,
inputs/marital-status?????????
A
native-country/?,
inputs/native-country?????????
9

occupation+?(
inputs/occupation?????????
-
race%?"
inputs/race?????????
=
relationship-?*
inputs/relationship?????????
+
sex$?!

inputs/sex?????????
7
	workclass*?'
inputs/workclass?????????
p 

 
? "%?"
?
0?????????
? ?
,__inference_functional_3_layer_call_fn_13510?_?\?Y?V?S?P?M?J?9:<=>????
???
???
$
age?
age?????????
6
capital-gain&?#
capital-gain?????????
6
capital-loss&?#
capital-loss?????????
0
	education#? 
	education?????????
8
education-num'?$
education-num?????????
*
fnlwgt ?
fnlwgt?????????
:
hours-per-week(?%
hours-per-week?????????
:
marital-status(?%
marital-status?????????
:
native-country(?%
native-country?????????
2

occupation$?!

occupation?????????
&
race?
race?????????
6
relationship&?#
relationship?????????
$
sex?
sex?????????
0
	workclass#? 
	workclass?????????
p

 
? "???????????
,__inference_functional_3_layer_call_fn_13635?_?\?Y?V?S?P?M?J?9:<=>????
???
???
$
age?
age?????????
6
capital-gain&?#
capital-gain?????????
6
capital-loss&?#
capital-loss?????????
0
	education#? 
	education?????????
8
education-num'?$
education-num?????????
*
fnlwgt ?
fnlwgt?????????
:
hours-per-week(?%
hours-per-week?????????
:
marital-status(?%
marital-status?????????
:
native-country(?%
native-country?????????
2

occupation$?!

occupation?????????
&
race?
race?????????
6
relationship&?#
relationship?????????
$
sex?
sex?????????
0
	workclass#? 
	workclass?????????
p 

 
? "???????????
,__inference_functional_3_layer_call_fn_14363?_?\?Y?V?S?P?M?J?9:<=>????
???
???
+
age$?!

inputs/age?????????
=
capital-gain-?*
inputs/capital-gain?????????
=
capital-loss-?*
inputs/capital-loss?????????
7
	education*?'
inputs/education?????????
?
education-num.?+
inputs/education-num?????????
1
fnlwgt'?$
inputs/fnlwgt?????????
A
hours-per-week/?,
inputs/hours-per-week?????????
A
marital-status/?,
inputs/marital-status?????????
A
native-country/?,
inputs/native-country?????????
9

occupation+?(
inputs/occupation?????????
-
race%?"
inputs/race?????????
=
relationship-?*
inputs/relationship?????????
+
sex$?!

inputs/sex?????????
7
	workclass*?'
inputs/workclass?????????
p

 
? "???????????
,__inference_functional_3_layer_call_fn_14425?_?\?Y?V?S?P?M?J?9:<=>????
???
???
+
age$?!

inputs/age?????????
=
capital-gain-?*
inputs/capital-gain?????????
=
capital-loss-?*
inputs/capital-loss?????????
7
	education*?'
inputs/education?????????
?
education-num.?+
inputs/education-num?????????
1
fnlwgt'?$
inputs/fnlwgt?????????
A
hours-per-week/?,
inputs/hours-per-week?????????
A
marital-status/?,
inputs/marital-status?????????
A
native-country/?,
inputs/native-country?????????
9

occupation+?(
inputs/occupation?????????
-
race%?"
inputs/race?????????
=
relationship-?*
inputs/relationship?????????
+
sex$?!

inputs/sex?????????
7
	workclass*?'
inputs/workclass?????????
p 

 
? "???????????
F__inference_predictions_layer_call_and_return_conditional_losses_15023\>?/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????
? ~
+__inference_predictions_layer_call_fn_15032O>?/?,
%?"
 ?
inputs?????????@
? "???????????
__inference_restore_fn_15179dJV?S
L?I
(?%
restored_tensors_0?????????
?
restored_tensors_1	
? "? ?
__inference_restore_fn_15206dMV?S
L?I
(?%
restored_tensors_0?????????
?
restored_tensors_1	
? "? ?
__inference_restore_fn_15233dPV?S
L?I
(?%
restored_tensors_0?????????
?
restored_tensors_1	
? "? ?
__inference_restore_fn_15260dSV?S
L?I
(?%
restored_tensors_0?????????
?
restored_tensors_1	
? "? ?
__inference_restore_fn_15287dVV?S
L?I
(?%
restored_tensors_0?????????
?
restored_tensors_1	
? "? ?
__inference_restore_fn_15314dYV?S
L?I
(?%
restored_tensors_0?????????
?
restored_tensors_1	
? "? ?
__inference_restore_fn_15341d\V?S
L?I
(?%
restored_tensors_0?????????
?
restored_tensors_1	
? "? ?
__inference_restore_fn_15368d_V?S
L?I
(?%
restored_tensors_0?????????
?
restored_tensors_1	
? "? ?
__inference_save_fn_15171?J&?#
?
?
checkpoint_key 
? "???
k?h

name?
0/name 
#

slice_spec?
0/slice_spec 
(
tensor?
0/tensor?????????
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_15198?M&?#
?
?
checkpoint_key 
? "???
k?h

name?
0/name 
#

slice_spec?
0/slice_spec 
(
tensor?
0/tensor?????????
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_15225?P&?#
?
?
checkpoint_key 
? "???
k?h

name?
0/name 
#

slice_spec?
0/slice_spec 
(
tensor?
0/tensor?????????
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_15252?S&?#
?
?
checkpoint_key 
? "???
k?h

name?
0/name 
#

slice_spec?
0/slice_spec 
(
tensor?
0/tensor?????????
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_15279?V&?#
?
?
checkpoint_key 
? "???
k?h

name?
0/name 
#

slice_spec?
0/slice_spec 
(
tensor?
0/tensor?????????
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_15306?Y&?#
?
?
checkpoint_key 
? "???
k?h

name?
0/name 
#

slice_spec?
0/slice_spec 
(
tensor?
0/tensor?????????
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_15333?\&?#
?
?
checkpoint_key 
? "???
k?h

name?
0/name 
#

slice_spec?
0/slice_spec 
(
tensor?
0/tensor?????????
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
__inference_save_fn_15360?_&?#
?
?
checkpoint_key 
? "???
k?h

name?
0/name 
#

slice_spec?
0/slice_spec 
(
tensor?
0/tensor?????????
`?]

name?
1/name 
#

slice_spec?
1/slice_spec 

tensor?
1/tensor	?
E__inference_sequential_layer_call_and_return_conditional_losses_13054k<=>?<?9
2?/
%?"
dense_input?????????|
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_13068k<=>?<?9
2?/
%?"
dense_input?????????|
p 

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_14902f<=>?7?4
-?*
 ?
inputs?????????|
p

 
? "%?"
?
0?????????
? ?
E__inference_sequential_layer_call_and_return_conditional_losses_14919f<=>?7?4
-?*
 ?
inputs?????????|
p 

 
? "%?"
?
0?????????
? ?
*__inference_sequential_layer_call_fn_13096^<=>?<?9
2?/
%?"
dense_input?????????|
p

 
? "???????????
*__inference_sequential_layer_call_fn_13123^<=>?<?9
2?/
%?"
dense_input?????????|
p 

 
? "???????????
*__inference_sequential_layer_call_fn_14932Y<=>?7?4
-?*
 ?
inputs?????????|
p

 
? "???????????
*__inference_sequential_layer_call_fn_14945Y<=>?7?4
-?*
 ?
inputs?????????|
p 

 
? "???????????
#__inference_signature_wrapper_13707?_?\?Y?V?S?P?M?J?9:<=>????
? 
???
$
age?
age?????????
6
capital-gain&?#
capital-gain?????????
6
capital-loss&?#
capital-loss?????????
0
	education#? 
	education?????????
8
education-num'?$
education-num?????????
*
fnlwgt ?
fnlwgt?????????
:
hours-per-week(?%
hours-per-week?????????
:
marital-status(?%
marital-status?????????
:
native-country(?%
native-country?????????
2

occupation$?!

occupation?????????
&
race?
race?????????
6
relationship&?#
relationship?????????
$
sex?
sex?????????
0
	workclass#? 
	workclass?????????"7?4
2

sequential$?!

sequential?????????