FROM python:3.13

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Work directory
WORKDIR /app

# Accept API keys as build or run args
ARG OPENAI_API_KEY
ARG ANTHROPIC_API_KEY
ARG GEMINI_API_KEY
ARG SEC_API_KEY
ARG SERPAPI_API_KEY

# Copy the project files into the container
COPY ./ /app/

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install faiss-cpu
RUN playwright install

# Set API keys as environment variables inside the container
ENV OPENAI_API_KEY=$OPENAI_API_KEY
ENV ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
ENV GEMINI_API_KEY=$GEMINI_API_KEY
ENV SEC_API_KEY=$SEC_API_KEY
ENV SERPAPI_API_KEY=$SERPAPI_API_KEY

# Expose the app port
EXPOSE 8000

# Run the server
CMD ["python", "-m", "llmvm.server.server"]
