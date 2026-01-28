FROM public.ecr.aws/lambda/python:3.13

RUN pip install onnxruntime keras-image-helper==0.0.2

ARG MODEL_NAME=finalChestXrayModel2.onnx
ARG MODEL_NAME_DATA=finalChestXrayModel2.onnx.data
ENV MODEL_NAME=${MODEL_NAME}

COPY ${MODEL_NAME} ${MODEL_NAME}
COPY ${MODEL_NAME_DATA} ${MODEL_NAME_DATA}

COPY lambda_function.py ./

CMD ["lambda_function.lambda_handler"]