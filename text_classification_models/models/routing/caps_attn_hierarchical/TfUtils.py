
import tensorflow as tf
from caps_attn_hierarchical import nest
import numpy as np

def mkMask(input_tensor, maxLen):
    shape_of_input = tf.shape(input_tensor)
    shape_of_output = tf.concat(axis=0, values=[shape_of_input, [maxLen]])

    oneDtensor = tf.reshape(input_tensor, shape=(-1,))
    flat_mask = tf.sequence_mask(oneDtensor, maxlen=maxLen)
    return tf.reshape(flat_mask, shape_of_output)


def reduce_avg(reduce_target, lengths, dim):
    """
    Args:
        reduce_target : shape(d_0, d_1,..,d_dim, .., d_k)
        lengths : shape(d0, .., d_(dim-1))
        dim : which dimension to average, should be a python number
    """
    shape_of_lengths = lengths.get_shape()
    shape_of_target = reduce_target.get_shape()
    if len(shape_of_lengths) != dim:
        raise ValueError(('Second input tensor should be rank %d, ' +
                         'while it got rank %d') % (dim, len(shape_of_lengths)))
    if len(shape_of_target) < dim+1 :
        raise ValueError(('First input tensor should be at least rank %d, ' +
                         'while it got rank %d') % (dim+1, len(shape_of_target)))

    rank_diff = len(shape_of_target) - len(shape_of_lengths) - 1
    mxlen = tf.shape(reduce_target)[dim]
    mask = mkMask(lengths, mxlen)
    if rank_diff!=0:
        len_shape = tf.concat(axis=0, values=[tf.shape(lengths), [1]*rank_diff])
        mask_shape = tf.concat(axis=0, values=[tf.shape(mask), [1]*rank_diff])
    else:
        len_shape = tf.shape(lengths)
        mask_shape = tf.shape(mask)
    lengths_reshape = tf.reshape(lengths, shape=len_shape)
    mask = tf.reshape(mask, shape=mask_shape)

    mask_target = reduce_target * tf.cast(mask, dtype=reduce_target.dtype)

    red_sum = tf.reduce_sum(mask_target, axis=[dim], keep_dims=False)
    red_avg = red_sum / (tf.to_float(lengths_reshape) + 1e-30)
    return red_avg


def reduce_sum(reduce_target, lengths, dim):
    """
    Args:
        reduce_target : shape(d_0, d_1,..,d_dim, .., d_k)
        lengths : shape(d0, .., d_(dim-1))
        dim : which dimension to average, should be a python number
    """
    shape_of_lengths = lengths.get_shape()
    shape_of_target = reduce_target.get_shape()
    if len(shape_of_lengths) != dim:
        raise ValueError(('Second input tensor should be rank %d, ' +
                         'while it got rank %d') % (dim, len(shape_of_lengths)))
    if len(shape_of_target) < dim+1 :
        raise ValueError(('First input tensor should be at least rank %d, ' +
                         'while it got rank %d') % (dim+1, len(shape_of_target)))

    rank_diff = len(shape_of_target) - len(shape_of_lengths) - 1
    mxlen = tf.shape(reduce_target)[dim]
    mask = mkMask(lengths, mxlen)
    if rank_diff!=0:
        len_shape = tf.concat(axis=0, values=[tf.shape(lengths), [1]*rank_diff])
        mask_shape = tf.concat(axis=0, values=[tf.shape(mask), [1]*rank_diff])
    else:
        len_shape = tf.shape(lengths)
        mask_shape = tf.shape(mask)
    lengths_reshape = tf.reshape(lengths, shape=len_shape)
    mask = tf.reshape(mask, shape=mask_shape)

    mask_target = reduce_target * tf.cast(mask, dtype=reduce_target.dtype)

    red_sum = tf.reduce_sum(mask_target, axis=[dim], keep_dims=False)

    return red_sum


def embed_lookup_last_dim(embedding, ids):
    '''
        embedding: shape(b_sz, tstp, emb_sz)
        ids : shape(b_sz, tstp)
    '''
    input_shape = tf.shape(embedding)
    time_steps = input_shape[0]
    def _create_ta(name, dtype):
        return tf.TensorArray(dtype=dtype,
                              size=time_steps,
                              tensor_array_name=name)
    input_ta = _create_ta('input_ta', embedding.dtype)
    fetch_ta = _create_ta('fetch_ta', ids.dtype)
    output_ta = _create_ta('output_ta', embedding.dtype)
    input_ta = input_ta.unpack(embedding)
    fetch_ta = fetch_ta.unpack(ids)

    def loop_body(time, output_ta):
        embed = input_ta.read(time) #shape(tstp, emb_sz) type of float32
        fetch_id = fetch_ta.read(time) #shape(tstp) type of int32
        out_emb = tf.nn.embedding_lookup(embed, fetch_id)
        output_ta = output_ta.write(time, out_emb)

        next_time = time+1
        return next_time, output_ta
    time = tf.constant(0)
    _, output_ta = tf.while_loop(cond=lambda time, *_: time < time_steps,
                  body=loop_body, loop_vars=(time, output_ta),
                  swap_memory=True)
    ret_t = output_ta.pack() #shape(b_sz, tstp, embd_sz)
    return ret_t


def entry_stop_gradients(target, mask):
    '''
    Args:
        target: a tensor
        mask: a boolean tensor that broadcast to the rank of that to target tensor
    Returns:
        ret: a tensor have the same value of target,
            but some entry will have no gradient during backprop
    '''
    mask_h = tf.logical_not(mask)

    mask = tf.cast(mask, dtype=target.dtype)
    mask_h = tf.cast(mask_h, dtype=target.dtype)
    ret = tf.stop_gradient(mask_h * target) + mask * target

    return ret


def last_dim_linear(inputs, output_size, bias, scope):
    '''
    Args:
        input: shape(b_sz, ..., rep_sz)
        output_size: a scalar, python number
    '''
    bias_start=0.0
    input_shape = tf.shape(inputs)
    out_shape = tf.concat(axis=0, values=[input_shape[:-1], [output_size]])
    input_size = int(inputs.get_shape()[-1])
    unbatch_input = tf.reshape(inputs, shape=[-1, input_size])

    unbatch_output = linear(unbatch_input, output_size, bias=bias,
                                            bias_start=bias_start, scope=scope)
    batch_output = tf.reshape(unbatch_output, shape=out_shape)

    return batch_output     # shape(b_sz, ..., output_size)


def linear(args, output_size, bias, bias_start=0.0, scope=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_start: starting value to initialize the bias; 0 by default.
    scope: (optional) Variable scope to create parameters in.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  with tf.variable_scope(scope or 'Linear') as outer_scope:
    weights = tf.get_variable(
        "weights", [total_arg_size, output_size], dtype=dtype)
    if len(args) == 1:
      res = tf.matmul(args[0], weights)
    else:
      res = tf.matmul(tf.concat(args, 1), weights)
    if not bias:
      return res
    with tf.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      biases = tf.get_variable(
          "biases", [output_size],
          dtype=dtype,
          initializer=tf.constant_initializer(bias_start, dtype=dtype))
  return tf.nn.bias_add(res, biases)


def masked_softmax(inp, seqLen):
    seqLen = tf.where(tf.equal(seqLen, 0), tf.ones_like(seqLen), seqLen)
    if len(inp.get_shape()) != len(seqLen.get_shape())+1:
        raise ValueError('rank of seqLen should be %d, but have the rank %d.\n'
                         % (len(inp.get_shape())-1, len(seqLen.get_shape())))
    mask = mkMask(seqLen, tf.shape(inp)[-1])
    masked_inp = tf.where(mask, inp, tf.ones_like(inp) * (-np.Inf))
    ret = tf.nn.softmax(masked_inp)
    return ret

from tensorflow.python.client import device_lib
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
