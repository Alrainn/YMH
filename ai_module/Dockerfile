# Build stage
FROM continuumio/miniconda3 as builder

# Copy environment file
COPY environment.yml /tmp/environment.yml

# Create conda environment and clean cache
RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Final stage
FROM continuumio/miniconda3

# Copy conda environment from builder
COPY --from=builder /opt/conda/envs/YMH_Projesi /opt/conda/envs/YMH_Projesi

# Create non-root user for the final image
RUN useradd -m -s /bin/bash appuser

# Set working directory for the application
WORKDIR /app

# Copy only necessary files
COPY --chown=appuser:appuser app.py ai.py Ilac.py requirements.txt ./
COPY --chown=appuser:appuser templates/ ./templates/
COPY --chown=appuser:appuser google_research/ ./google_research/

# Switch to the non-root user
USER appuser

# Set the shell for subsequent commands to run within the conda environment
SHELL ["conda", "run", "-n", "YMH_Projesi", "/bin/bash", "-c"]

# Expose the port where the application will listen
EXPOSE 5000/tcp

# Set environment variables
ENV FLASK_ENV=production
ENV FLASK_APP=app.py
# Python çıktısının arabelleğe alınmamasını sağlar, logları hemen görürsün:
ENV PYTHONUNBUFFERED=1
# .pyc dosyalarının oluşturulmasını engeller, imajı temiz tutar:
ENV PYTHONDONTWRITEBYTECODE=1
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000 

# Add a healthcheck to verify the application is running
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run the application using gunicorn within the specified Conda environment
CMD ["conda", "run", "--no-capture-output", "-n", "YMH_Projesi", "gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]