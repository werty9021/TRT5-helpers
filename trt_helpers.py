from keras.models import load_model
import keras.backend as K
import tensorflow as tf

import pycuda.driver as cuda
# This import causes pycuda to automatically manage CUDA context creation and cleanup.
import pycuda.autoinit

import tensorrt as trt
import uff

import numpy as np


def GiB(val):
    return val * 1 << 30


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class Model:
    def __init__(self, parser):
        self.parser = parser  # This object should continue to live, otherwise Network output is NaN

    def parse(self, network):
        raise NotImplementedError

    def __del__(self):
        del self.parser


class UFFModel(Model):
    def __init__(self, file_path, input_name, input_shape, output_name, uff_buffer=None):
        super().__init__(trt.UffParser())

        self.file_path = file_path
        self.input_name = input_name
        self.input_shape = input_shape
        self.output_name = output_name
        self.uff_buffer = uff_buffer

    def parse(self, network):
        # Parse the Uff Network
        # Register in- and outputs
        self.parser.register_input(self.input_name, self.input_shape)
        self.parser.register_output(self.output_name)
        # Parse from a buffer or let parser read a file
        if self.uff_buffer:
            self.parser.parse_buffer(self.uff_buffer, network)
        else:
            self.parser.parse(self.file_path, network)


class TFModel(UFFModel):
    def __init__(self, file_path, input_name, input_shape, output_name):
        uff_buffer = uff.from_tensorflow_frozen_model(file_path)
        super().__init__(file_path, input_name, input_shape, output_name, uff_buffer)


class KerasModel(UFFModel):
    def __init__(self, file_path, input_shape):
        # Stop TF from occupying the whole GPU for nothing.
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        K.set_session(sess)

        K.set_learning_phase(0)
        model = load_model(file_path, compile=False)
        K.set_learning_phase(0)
        output_name = model.output.op.name
        input_name = model.input.op.name
        frozen_graph = tf.graph_util.remove_training_nodes(
            tf.graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), [output_name]))

        # Convert Tensorflow frozen graph to UFF file
        uff_buffer = uff.from_tensorflow(frozen_graph, output_file=file_path.replace('.h5', '.uff'))
        super().__init__(file_path, input_name, input_shape, output_name, uff_buffer)


class TrtEngine:
    def __init__(self, model, max_batch_size=1):
        self._logger = trt.Logger(trt.Logger.WARNING)
        self.max_batch_size = max_batch_size
        self._engine = self._build_engine(model)

    def _build_engine(self, model):
        with trt.Builder(self._logger) as builder, builder.create_network() as network:
            builder.max_workspace_size = GiB(1)
            #builder.fp16_mode = True
            builder.max_batch_size = self.max_batch_size
            # Parse the model to populate the TRT network
            model.parse(network)
            # Build and return an engine.
            return builder.build_cuda_engine(network)

    def _allocate_buffers(self, batch_size):
        inputs, outputs, bindings, stream = [], [], [], cuda.Stream()
        for binding in self._engine:
            size = batch_size * trt.volume(self._engine.get_binding_shape(binding))
            dtype = trt.nptype(self._engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if self._engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs, bindings, stream

    def infer(self, input_data):
        # Final list to hold the inference results
        output_data = []

        # Cuda stream inventory
        streams = []
        # Per stream outputs
        streams_outputs = []

        # Split on max batch size
        for i in range(0, input_data.shape[0], self.max_batch_size):
            # Get data and the resulting size
            batch_data = input_data[i:i+self.max_batch_size]
            batch_size = batch_data.shape[0]

            # Allocate memory for this batch
            inputs, outputs, bindings, stream = self._allocate_buffers(batch_size)
            # Store the respective Cuda stream and its outputs
            streams.append(stream)
            streams_outputs.append((outputs, batch_size))

            # Copy input data into pagelocked host mem
            np.copyto(inputs[0].host, batch_data.ravel())

            # Create Execution Context
            with self._engine.create_execution_context() as context:
                # This is generalized for multiple inputs/outputs.
                # inputs and outputs are expected to be lists of HostDeviceMem objects.
                # Transfer input data to the GPU.
                [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
                # Run inference.
                context.execute_async(bindings=bindings, stream_handle=stream.handle, batch_size=batch_size)
                # Transfer predictions back from the GPU.
                [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]

        # Synchronize the stream (i.e. wait for it to be finished)
        [stream.synchronize() for stream in streams]
        # For each stream extract the outputs
        for (outputs, batch_size) in streams_outputs:
            # Return only the host outputs.
            # Numpy split to split results from a single batch into separate arrays
            output_data.extend(*[np.split(out.host, batch_size) for out in outputs])

        # Return the inference results
        return output_data

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def __del__(self):
        del self._engine


if __name__ == "__main__":
    import os.path
    model = KerasModel(
        file_path=os.path.join(os.path.dirname(__file__), "models/dncnn.h5"),
        input_shape=(1, 512, 512),
    )
    inference_engine = TrtEngine(model, max_batch_size=16)

    data = np.load(os.path.join(os.path.dirname(__file__), 'data.npy'))
    prediction = inference_engine.infer(data)
    print(prediction)
