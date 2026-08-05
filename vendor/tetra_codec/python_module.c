#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

#include "tetra-codec.h"

#define PCM_SAMPLES_PER_FRAME 240
#define CODED_BYTES_PER_FRAME 18

static PyObject *encode_stream(PyObject *self, PyObject *args)
{
    const char *pcm_bytes;
    Py_ssize_t pcm_size;
    Py_ssize_t frame_bytes = PCM_SAMPLES_PER_FRAME * (Py_ssize_t)sizeof(int16_t);
    Py_ssize_t frame_count;
    PyObject *result;
    tetra_codec *encoder;
    Py_ssize_t frame;

    (void)self;
    if (!PyArg_ParseTuple(args, "y#", &pcm_bytes, &pcm_size)) {
        return NULL;
    }
    if (pcm_size == 0 || pcm_size % frame_bytes != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "PCM data must contain complete 240-sample int16 frames");
        return NULL;
    }

    frame_count = pcm_size / frame_bytes;
    result = PyBytes_FromStringAndSize(NULL, frame_count * CODED_BYTES_PER_FRAME);
    if (result == NULL) {
        return NULL;
    }
    encoder = tetra_encoder_create();
    if (encoder == NULL) {
        Py_DECREF(result);
        return PyErr_NoMemory();
    }

    for (frame = 0; frame < frame_count; ++frame) {
        uint8_t *coded = (uint8_t *)PyBytes_AS_STRING(result)
                         + frame * CODED_BYTES_PER_FRAME;
        memset(coded, 0, CODED_BYTES_PER_FRAME);
        tetra_encode(encoder,
                     (const int16_t *)(pcm_bytes + frame * frame_bytes),
                     coded);
    }
    tetra_codec_destroy(encoder);
    return result;
}

static PyObject *decode_stream(PyObject *self, PyObject *args)
{
    const char *coded_bytes;
    Py_ssize_t coded_size;
    Py_ssize_t frame_count;
    Py_ssize_t pcm_size;
    PyObject *result;
    tetra_codec *decoder;
    Py_ssize_t frame;

    (void)self;
    if (!PyArg_ParseTuple(args, "y#", &coded_bytes, &coded_size)) {
        return NULL;
    }
    if (coded_size == 0 || coded_size % CODED_BYTES_PER_FRAME != 0) {
        PyErr_SetString(PyExc_ValueError,
                        "coded data must contain complete 18-byte ACELP frames");
        return NULL;
    }

    frame_count = coded_size / CODED_BYTES_PER_FRAME;
    pcm_size = frame_count * PCM_SAMPLES_PER_FRAME * (Py_ssize_t)sizeof(int16_t);
    result = PyBytes_FromStringAndSize(NULL, pcm_size);
    if (result == NULL) {
        return NULL;
    }
    decoder = tetra_decoder_create();
    if (decoder == NULL) {
        Py_DECREF(result);
        return PyErr_NoMemory();
    }

    for (frame = 0; frame < frame_count; ++frame) {
        tetra_decode(decoder,
                     (const uint8_t *)coded_bytes + frame * CODED_BYTES_PER_FRAME,
                     (int16_t *)PyBytes_AS_STRING(result)
                         + frame * PCM_SAMPLES_PER_FRAME,
                     0);
    }
    tetra_codec_destroy(decoder);
    return result;
}

static PyMethodDef module_methods[] = {
    {"encode_stream", encode_stream, METH_VARARGS,
     "Encode complete 240-sample PCM frames with one fresh codec state."},
    {"decode_stream", decode_stream, METH_VARARGS,
     "Decode complete 18-byte ACELP frames with one fresh codec state."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_tetra_acelp",
    "Native ETSI TETRA ACELP codec adapter.",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit__tetra_acelp(void)
{
    return PyModule_Create(&module_definition);
}
