	Ϡ�R@Ϡ�R@!Ϡ�R@	�[pb�?�[pb�?!�[pb�?"�
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetailsLϠ�R@�:���?1�25	ސ�?ARf`X�?I�3�?Oc@Y噗��;�?rEagerKernelExecute 0*	�$���Y@2U
Iterator::Model::ParallelMapV2CF�7��?!ǽSgW:@)CF�7��?1ǽSgW:@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat/��0�?!�0�7@)����ޓ?1�LG#�3@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice)x
�Rϒ?!f�����1@))x
�Rϒ?1f�����1@:Preprocessing2F
Iterator::Model3SZK �?!P���zE@)#�ng_y�?1(�o(�/@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip8J^�c@�?!�7��L@)��Ր�ǂ?1��dt�!@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMape�fb��?!;���9@)x
�Rς�?1Rk��,�@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorA,�9$�p?!�AF��@)A,�9$�p?1�AF��@:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is POTENTIALLY input-bound because 21.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).high"�53.1 % of the total step time sampled is spent on 'Kernel Launch'. It could be due to CPU contention with tf.data. In this case, you may try to set the environment variable TF_GPU_THREAD_MODE=gpu_private.*no9�[pb�?I�� S@Qj�lF�6@Zno>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	�:���?�:���?!�:���?      ��!       "	�25	ސ�?�25	ސ�?!�25	ސ�?*      ��!       2	Rf`X�?Rf`X�?!Rf`X�?:	�3�?Oc@�3�?Oc@!�3�?Oc@B      ��!       J	噗��;�?噗��;�?!噗��;�?R      ��!       Z	噗��;�?噗��;�?!噗��;�?b      ��!       JGPUY�[pb�?b q�� S@yj�lF�6@