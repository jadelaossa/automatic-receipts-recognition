FROM public.ecr.aws/lambda/python:3.10

# Install Tesseract-OCR and any dependencies
RUN curl -o /etc/yum.repos.d/tesseract.repo \
    https://download.opensuse.org/repositories/home:/Alexander_Pozdnyakov/CentOS_7/home:Alexander_Pozdnyakov.repo && \
    yum update -y && \
    yum install -y tesseract tesseract-langpack-spa && \
    yum clean all

# Install OpenCV dependencies
RUN yum install -y \
    libSM \
    libXrender \
    libXext \
    libglvnd-devel \
    mesa-libGL \
    && yum clean all

# Copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the specified packages
RUN pip install -r requirements.txt

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler (could also be done as a parameter override outside of the Dockerfile)
CMD [ "app.handler" ]