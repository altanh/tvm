# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Threefry PRNG with splitting based on
- J. K. Salmon, M. A. Moraes, R. O. Dror and D. E. Shaw, "Parallel random numbers: As easy as 1, 2,
  3," SC '11: Proceedings of 2011 International Conference for High Performance Computing,
  Networking, Storage and Analysis, Seattle, WA, 2011, pp. 1-12, doi: 10.1145/2063384.2063405.
- Claessen, K. ; Palka, M. (2013) "Splittable Pseudorandom Number Generators using Cryptographic
  Hashing". Proceedings of Haskell Symposium 2013 pp. 47-58.  MLA
- Ferguson, Niels, et al. "The Skein hash function family." Submission to NIST (round 3) 7.7.5
  (2010): 3.


Threefry is a counter based PRNG: given a unique input, it generates a unique random number. As
there is no state to maintain, we can apply it to a sequence of numbers (0..N) to generate a
sequence of random numbers in parallel. In order to make the PRNG splittable (that is we can
generate a sequence of random numbers in one place, and another sequence in another), we add a path
and key in addition to the counter. The path allows us to encode a sequence of splits (a 0 in the
path indicates the left result of a split, a 1 indicates the right). To avoid continuously growing
the path, we can compress an existing path into the key portion of the generator by hashing the
current key, path, and counter to create the new key (this same technique is used if we run out of
room for the counter).

This module use encoding e4 from the appendix of "Splittable Pseudorandom Number Generators using
Cryptographic Hashing" (confusingly, the definition in the paper uses e3 to define the encoding
function). This encoding uses a 10 element uint64 tensor where each byte has the following meaning:

.. code-block:

    gen:
    words: 0 1 2 3 | 4 5  | 6 7     | 8 9
    usage: key     | path | counter | position of next step in path encoded in binary
                                      ex: 0b00010 -> next path entry goes one from the right

Right now, counter only uses the rightmost word.
"""
import tvm
import tvm.topi
from ... import tir
from ...tir import ir_builder

# Threefry rotation constants from the Skein paper ("The Skein Hash Function Family"
# https://www.schneier.com/wp-content/uploads/2015/01/skein.pdf)
_ROTATIONS = {
    4: [[14, 16], [52, 57], [23, 40], [5, 37], [25, 33], [46, 12], [58, 22], [32, 32]],
    8: [
        [46, 36, 19, 37],
        [33, 27, 14, 42],
        [17, 49, 36, 39],
        [44, 9, 54, 56],
        [39, 30, 34, 24],
        [13, 50, 10, 17],
        [25, 29, 39, 43],
        [8, 35, 56, 22],
    ],
    16: [
        [24, 13, 8, 47, 8, 17, 22, 37],
        [38, 19, 10, 55, 49, 18, 23, 52],
        [33, 4, 51, 13, 34, 41, 59, 17],
        [5, 20, 48, 41, 47, 28, 16, 25],
        [41, 9, 37, 31, 12, 47, 44, 30],
        [16, 34, 56, 51, 4, 53, 42, 41],
        [31, 44, 47, 46, 19, 42, 44, 25],
        [9, 48, 35, 52, 23, 31, 37, 20],
    ],
}

# Threefry permutation constants from the Skein paper ("The Skein Hash Function Family"
# https://www.schneier.com/wp-content/uploads/2015/01/skein.pdf)
_PERMUTATIONS = {
    4: [0, 3, 2, 1],
    8: [2, 1, 4, 7, 6, 5, 0, 3],
    16: [0, 9, 2, 13, 6, 11, 4, 15, 10, 7, 12, 3, 14, 5, 8, 1],
}


def _threefry(
    irb, key_buf, key_offset, counter_buf, counter_offset, out_buf, out_offset, out_shape
):
    """IRBuilder code for running Threefry

    Parameters
    ----------
    irb: IRBuilder
        IRBuilder that this code will be generated for.

    key_buf: BufferVar
        Buffer to read the key from.

    key_offset: number
        Threefry will write to :code:`key_buf[key_offset:key_offset+4]`

    counter_buf: BufferVar
        Buffer to read the counter from.

    counter_offset: number
        Threefry will write to :code:`counter_buf[counter_offset:counter_offset+4]`

    out_buf: BufferVar
        Buffer to read the counter from.

    counter_offset: number
        Threefry will write to :code:`out_buf[out_offset:out_offset+4*product(out_shape)]`

    out_shape: number
        Determines the number of ouput states to generate. :code:`state[i]` will correspond to counter+i.
    """
    nrounds = 20
    nwords = 4
    iwidth = 64
    assert nrounds % 4 == 0
    assert nwords in [4, 8, 16]

    assert key_buf.dtype == "uint64"  # TODO: support 32 bit inputs
    assert key_buf.dtype == counter_buf.dtype

    def mix(a, b, rotation):
        x = a + b  # TODO should be wrapping
        y = x ^ ((b << rotation) | (b >> (iwidth - rotation)))
        return [x, y]

    # temporary buffer for holding the results of _PERMUTATIONS
    tmp = irb.allocate(out_buf.dtype, out_shape, name="tmp", scope="global")
    tmp_offset = 0

    # Initialize entire key. It is composed of the original key with one
    # element appended. The appended element is the xor of all key words plus a
    # constant.
    full_key = irb.allocate("uint64", nwords + 1, name="full_key", scope="global")
    for i in range(nwords):
        full_key[i] = key_buf[key_offset + i]
    # initial key constant, full_key[nwords] is equivalent to k_{N_W} in the Skein paper.
    full_key[nwords] = tvm.tir.const(0x1BD11BDAA9FC1A22, dtype="uint64")
    for i in range(nwords):
        full_key[nwords] ^= key_buf[key_offset + i]  # TODO: wrapping

    # TODO: overwrite counter instead?
    with irb.for_range(0, out_shape, dtype="uint64", name="i") as i:
        for j in range(nwords):
            out_buf[out_offset + i * nwords + j] = counter_buf[counter_offset + j] + i

    def key_schedule(s, i):
        # Threefry uses no tweak, so the key schedule is simple
        if i == nwords - 1:
            return full_key[(s + i) % (nwords + 1)] + tvm.tir.const(s, dtype="uint64")
        return full_key[(s + i) % (nwords + 1)]

    with irb.for_range(0, out_shape, name="l") as l:  # pylint: disable=invalid-name
        for i in range(nrounds // 4):
            for j in range(nwords):
                out_buf[out_offset + l * nwords + j] += key_schedule(i, j)  # TODO wrapping
            for k in range(4):
                for j in range(nwords // 2):
                    (
                        out_buf[out_offset + l * nwords + j * 2 + 0],
                        out_buf[out_offset + l * nwords + j * 2 + 1],
                    ) = mix(
                        out_buf[out_offset + l * nwords + j * 2 + 0],
                        out_buf[out_offset + l * nwords + j * 2 + 1],
                        _ROTATIONS[nwords][(i * 4 + k) % 8][j],
                    )
                for j in range(nwords):
                    tmp[tmp_offset + l * nwords + j] = out_buf[
                        out_offset + l * nwords + _PERMUTATIONS[nwords][j]
                    ]
                # number of rounds is even, so out always contains the result
                (out_buf, tmp) = (tmp, out_buf)
                (out_offset, tmp_offset) = (tmp_offset, out_offset)


def threefry_generate(gen, out_shape):
    """Generate a series of random values

    Notes
    -----
    This function uses the counter portion of the generator state to generate a series of random
    numbers in parallel. Random number `i` is generated by applying Threefry to the current
    generator state with the counter portion incremented by `i`. This means that each random number
    is generated independently from each other random number, so we can compute them in parallel.

    If there is not enough room left in the counter to generate the desired shape of random values,
    then a new generator is created by applying Threefry to the current key, path, and counter.
    This new generator will have a reset counter.

    Parameters
    ----------
    gen : Tensor[10, uint64]
        Generator state. Can be create with :py:func:`tvm.relay.threefry_seed`. This should not be used in
        another function, otherwise random numbers will be repeated.

    out_shape : Sequence[int]
        Output shape of the random numbers. Product of all dimensions must be a multiple of 4.

    Returns
    -------
    rand : Tensor[out_shape, uint64]
        Tensor of random numbers with shape `out_shape`.
    """
    out_len = 1
    for s in out_shape:
        out_len *= s
    assert (
        out_len.value % 4 == 0
    ), f"Threefry can only generate arrays who's size is a multiple of 4 ({out_len} was provided)."
    assert (
        out_len.value <= 2 ** 64 - 1
    ), f"Can only generate up to 2^64 random numbers, but {out_len} were requested."

    def gen_ir(gen_ptr, out_gen_ptr, out_array_ptr):
        irb = ir_builder.create()
        gen = irb.buffer_ptr(gen_ptr)
        out_gen = irb.buffer_ptr(out_gen_ptr)
        out_array = irb.buffer_ptr(out_array_ptr)

        # Create a temporary array to hold the generator state we will use to create the random
        # numbers. We cannot use gen because we may need to update the key + path if there is not
        # enough room in the counter.
        tmp = irb.allocate(gen.dtype, 10, name="tmp", scope="global")

        # TODO(tkonolige): for now we only use the last word of the counter for counting. Its too
        # much work to figure out how to do 128 bit addition.

        # Max value for counter should be 2**64-2 because we need to reserve a special value to
        # indicate the counter is used up.
        with irb.if_scope(gen[7] < tir.const(2 ** 64 - 1, dtype=gen.dtype) - out_len):
            for i in range(10):
                tmp[i] = gen[i]
        with irb.else_scope():
            # no room left in the counter, we have to change the path or key
            with irb.if_scope(gen[8] == 0 and gen[9] == 0):
                # out of room in the path, have to generate new key

                # The paper says the counter that we will be hashing should be a special value of
                # all ones. We need to allocate some space for it because we cannot overwrite gen.
                tmp_counter = irb.allocate(gen.dtype, 2, name="tmp_counter", scope="global")
                tmp_counter[0] = tir.const(0xFFFFFFFFFFFFFFFF, dtype=gen.dtype)
                tmp_counter[1] = tir.const(0xFFFFFFFFFFFFFFFF, dtype=gen.dtype)
                _threefry(irb, gen, 0, tmp_counter, 0, tmp, 0, 1)
                tmp[4] = tir.const(0, dtype=gen.dtype)  # zero path, i.e. no path
                tmp[5] = tir.const(0, dtype=gen.dtype)
                tmp[6] = tir.const(0, dtype=gen.dtype)  # zero counter
                tmp[7] = tir.const(0, dtype=gen.dtype)
                tmp[8] = tir.const(1 << 63, dtype=gen.dtype)  # one in the leftmost position
                tmp[9] = tir.const(0, dtype=gen.dtype)
            with irb.else_scope():
                tmp[0] = gen[0]
                tmp[1] = gen[1]
                tmp[2] = gen[2]
                tmp[3] = gen[3]
                tmp[4] = gen[4] | gen[8]  # add a 1 to the path
                tmp[5] = gen[5] | gen[9]
                tmp[6] = tir.const(0, dtype=gen.dtype)  # zero counter
                tmp[7] = tir.const(0, dtype=gen.dtype)
                _shift_right(irb, gen[8], gen[9], tmp, 8, tmp, 9)

        # Compute random values
        _threefry(irb, tmp, 0, tmp, 4, out_array, 0, out_len // 4)

        # Update generator state
        out_gen[0] = tmp[0]  # key stays the same
        out_gen[1] = tmp[1]
        out_gen[2] = tmp[2]
        out_gen[3] = tmp[3]
        out_gen[4] = tmp[4]  # path stays the same
        out_gen[5] = tmp[5]
        out_gen[6] = tir.const(0, dtype=gen.dtype)  # unused, leave it as 0
        out_gen[7] = tmp[7] + tir.Cast(gen.dtype, out_len)  # increment counter
        out_gen[8] = tmp[8]  # path unchanged, so no update here

        return irb.get()

    out_gen = tvm.tir.decl_buffer((10,), name="out_gen", dtype="uint64")
    out_array = tvm.tir.decl_buffer(out_shape, name="out_array", dtype="uint64")
    return tvm.te.extern(
        [out_gen.shape, out_array.shape],
        [gen],
        lambda ins, outs: gen_ir(ins[0], outs[0], outs[1]),
        out_buffers=[out_gen, out_array],
        name="threefry_generate",
        tag="threefry_generate",
    )


def _shift_right(irb, a, b, out_a, a_off, out_b, b_off):
    """Shift a 128bit number composed of two 64 bit words right by one"""
    with irb.if_scope(a == 1):
        out_a[a_off] = tir.const(0, dtype=a.dtype)
        out_b[b_off] = tir.const(0x8000000000000000, dtype=a.dtype)
    with irb.else_scope():
        with irb.if_scope(a == 0):
            out_a[a_off] = tir.const(0, dtype=a.dtype)
            out_b[b_off] = b >> 1
        with irb.else_scope():
            out_a[a_off] = a >> 1
            out_b[b_off] = tir.const(0, dtype=a.dtype)


def threefry_split(gen):
    """Split a single generator state into two new ones

    Notes
    -----
    The new generator is created by appending a one (for the right output) or a zero (for the left
    output) to the end of the path portion of the generator If there is no longer and room in the
    path, then we create a new key portion of the generator by applying Threefry to the old state,
    path, and counter. i.e. :code:`new_key = threefry(old_key, [old_path, old_counter])`. This
    resets the path portion of the new generator.

    Parameters
    ----------
    gen : Tensor[10, uint64]
        Generator state. Can be create with :py:func:`tvm.relay.threefry_seed`. This should not be used in
        another function, otherwise random numbers will be repeated.

    Returns
    -------
    out_gen_left : Tensor[10, uint64]
        New generator state that is distinct from `out_gen_right`.

    out_gen_right : Tensor[10, uint64]
        New generator state that is distinct from `out_gen_left`.
    """

    def gen_ir(gen_ptr, out_left_ptr, out_right_ptr):
        irb = ir_builder.create()
        gen = irb.buffer_ptr(gen_ptr)
        out_left = irb.buffer_ptr(out_left_ptr)
        out_right = irb.buffer_ptr(out_right_ptr)

        with irb.if_scope(gen[8] == 0 and gen[9] == 0):
            # Generate new key because we have run out of room to extend the path
            _threefry(irb, gen, 0, gen, 4, out_left, 0, 1)
            out_left[4] = tir.const(0, dtype=gen.dtype)
            out_left[5] = tir.const(0, dtype=gen.dtype)
            out_left[6] = tir.const(0, dtype=gen.dtype)  # counter gets zeroed
            out_left[7] = tir.const(0, dtype=gen.dtype)  # counter gets zeroed
            out_left[8] = tir.const(
                1 << 62, dtype=gen.dtype
            )  # one in the second from the leftmost position
            out_left[9] = tir.const(0, dtype=gen.dtype)

            out_right[0] = out_left[0]
            out_right[1] = out_left[1]
            out_right[2] = out_left[2]
            out_right[3] = out_left[3]
            out_right[4] = tir.const(1 << 63, dtype=gen.dtype)  # one in the leftmost position
            out_right[5] = tir.const(0, dtype=gen.dtype)
            out_right[6] = tir.const(0, dtype=gen.dtype)
            out_right[7] = tir.const(0, dtype=gen.dtype)
            out_right[8] = tir.const(
                1 << 62, dtype=gen.dtype
            )  # one in the second from the leftmost position
            out_right[9] = tir.const(0, dtype=gen.dtype)
        with irb.else_scope():
            out_left[0] = gen[0]
            out_left[1] = gen[1]
            out_left[2] = gen[2]
            out_left[3] = gen[3]
            out_left[4] = gen[4]  # adding a zero here, but its already zero padded
            out_left[5] = gen[5]
            out_left[6] = gen[6]
            out_left[7] = gen[7]
            # move path position over one bit
            _shift_right(irb, gen[8], gen[9], out_left, 8, out_left, 9)

            out_right[0] = gen[0]
            out_right[1] = gen[1]
            out_right[2] = gen[2]
            out_right[3] = gen[3]
            out_right[4] = gen[4] | gen[8]  # add a one to the path
            out_right[5] = gen[5] | gen[9]
            out_right[6] = gen[6]
            out_right[7] = gen[7]
            _shift_right(irb, gen[8], gen[9], out_right, 8, out_right, 9)

        return irb.get()

    out_left = tvm.tir.decl_buffer((10,), name="out_left", dtype="uint64")
    out_right = tvm.tir.decl_buffer((10,), name="out_right", dtype="uint64")
    return tvm.te.extern(
        [out_left.shape, out_right.shape],
        [gen],
        lambda ins, outs: gen_ir(ins[0], outs[0], outs[1]),
        out_buffers=[out_left, out_right],
        name="threefry_split",
        tag="threefry_split",
    )
