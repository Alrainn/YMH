# Build stage
FROM continuumio/miniconda3 as builder

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment
RUN conda env create -f /tmp/environment.yml

# Final stage
FROM continuumio/miniconda3

# Copy conda environment from builder
COPY --from=builder /opt/conda/envs/YMH_Projesi /opt/conda/envs/YMH_Projesi

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY --chown=appuser:appuser app.py ai.py Ilac.py requirements.txt ./
COPY --chown=appuser:appuser templates/ ./templates/
COPY --chown=appuser:appuser google_research/ ./google_research/

# Switch to non-root user
USER appuser

# Activate conda environment
SHELL ["conda", "run", "-n", "YMH_Projesi", "/bin/bash", "-c"]

# Expose port with description
EXPOSE 5000/tcp

# Set environment variables
ENV FLASK_ENV=production
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Set memory limits
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Run the application with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
