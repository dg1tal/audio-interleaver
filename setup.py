from setuptools import Extension, setup


setup(
    ext_modules=[
        Extension(
            "audio_interleaver._tetra_acelp",
            sources=[
                "vendor/tetra_codec/python_module.c",
                "vendor/tetra_codec/source/tetra-codec.c",
                "vendor/tetra_codec/source/tetra-codec-impl.c",
            ],
            include_dirs=[
                "vendor/tetra_codec/include",
                "vendor/tetra_codec/source",
            ],
        )
    ]
)
