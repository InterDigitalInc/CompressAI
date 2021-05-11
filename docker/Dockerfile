ARG PYTORCH_IMAGE
FROM ${PYTORCH_IMAGE}-devel as builder

WORKDIR /tmp/compressai
COPY compressai.tar.gz .
RUN pip install pybind11
RUN tar xzf compressai.tar.gz && \
		python3 setup.py bdist_wheel

FROM ${PYTORCH_IMAGE}-runtime

LABEL maintainer="compressai@interdigital.com"

WORKDIR /tmp
COPY --from=builder /tmp/compressai/dist/compressai-*.whl .
RUN pip install compressai-*.whl && \
		python3 -c 'import compressai'

# Install jupyter?
ARG WITH_JUPYTER=0
RUN if [ "$WITH_JUPYTER" = "1" ]; then \
		pip3 install jupyter ipywidgets && \
		jupyter nbextension enable --py widgetsnbextension \
		; fi

WORKDIR /workspace
CMD ["bash"]
