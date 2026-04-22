# Windows Setup Guide for CrowdShield AI

This guide provides step-by-step instructions for setting up and running CrowdShield AI on Windows.

## Prerequisites

- **Windows 10/11** (Pro, Enterprise, or Home with WSL2)
- **Docker Desktop for Windows** (with WSL2 backend)
- **Git for Windows** (or any Git client)
- **Python 3.11+** (optional, for local testing outside Docker)
- **Node.js 20+** (optional, for local frontend development)

## Installation Steps

### 1. Install Docker Desktop

1. Download **Docker Desktop for Windows** from [docker.com](https://www.docker.com/products/docker-desktop)
2. Install and follow the setup wizard
3. Ensure **WSL2 backend** is enabled:
   - Open Docker Desktop → Settings → General
   - Check "Use the WSL 2 based engine"
4. Verify installation in PowerShell:
   ```powershell
   docker --version
   docker-compose --version
   ```

### 2. Clone/Extract the Repository

```powershell
# If you have a ZIP file
Expand-Archive -Path "CrowdShield_AI.zip" -DestinationPath "C:\Dev"

# Or clone from Git
git clone <repository-url>
cd CrowdShield_AI
```

### 3. Configure Environment Variables

1. Create a `.env` file in the project root (use `.env.example` as a template if available):

```powershell
# Create .env file
New-Item -Path ".env" -ItemType File

# Edit the file with your settings (use Notepad or your preferred editor)
notepad .env
```

2. Add the following environment variables:

```
POSTGRES_DB=crowdshield
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
REDIS_URL=redis://redis:6379/0
BEHAVIOR_WINDOW_SIZE=30
ALERT_PERSISTENCE_FRAMES=60
```

### 4. Build and Start Services

```powershell
# Navigate to project directory
cd CrowdShield_AI

# Build Docker images
docker-compose build

# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# View logs (optional)
docker-compose logs -f
```

### 5. Access the Application

- **Frontend Dashboard**: http://localhost:3000
- **Frontend Dev (Vite)**: http://localhost:5173
- **API Gateway**: http://localhost:8000
- **Swagger/API Docs**: http://localhost:8000/docs
- **Individual Services**:
  - Ingestion Service: http://localhost:8001
  - Vision Service: http://localhost:8002
  - Pose Service: http://localhost:8003
  - Behavior Service: http://localhost:8004
  - Alert Service: http://localhost:8005

## Troubleshooting

### Services Won't Start

1. Check Docker Desktop is running
2. Verify WSL2 is enabled:
   ```powershell
   wsl -l -v
   ```
3. Check available ports aren't in use:
   ```powershell
   netstat -ano | findstr :8000
   ```
4. View detailed logs:
   ```powershell
   docker-compose logs <service-name>
   ```

### WSL2 Backend Issues

If Docker fails with WSL2 issues:

1. Update WSL2:
   ```powershell
   wsl --update
   ```
2. Reinstall WSL2:
   ```powershell
   wsl --install
   ```
3. Set WSL2 as default:
   ```powershell
   wsl --set-default-version 2
   ```

### Port Already in Use

Change ports in `docker-compose.yml`:

```yaml
ports:
  - "8001:8000"  # Change first number to unused port, e.g., "9001:8000"
```

### Memory Issues

If containers crash due to memory:

1. Increase Docker memory allocation:
   - Docker Desktop → Settings → Resources
   - Set "Memory" to at least 4GB (8GB recommended)

### Database Connection Issues

Verify PostgreSQL is running:

```powershell
docker-compose logs postgres
```

Reset the database if needed:

```powershell
docker-compose down -v  # Remove volumes
docker-compose up -d    # Recreate fresh
```

## Local Development (Without Docker)

### Python Services

1. Install Python 3.11+
2. Create virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Run individual service:
   ```powershell
   cd backend/vision-service
   uvicorn app.main:app --reload
   ```

### Frontend

1. Install Node.js 20+
2. Install dependencies:
   ```powershell
   cd frontend
   npm install
   ```
3. Start development server:
   ```powershell
   npm run dev
   ```

## Useful Commands

```powershell
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Stop and remove all containers + volumes
docker-compose down -v

# Rebuild specific service
docker-compose build <service-name>

# Access service shell
docker-compose exec <service-name> /bin/bash

# Check service status
docker-compose ps
```

## Performance Tips

1. **Enable hardware acceleration**: Docker Desktop → Settings → Docker Engine
2. **Optimize WSL2 memory**: Adjust in `.wslconfig` if needed
3. **Use SSD**: Store project on solid-state drive for better I/O
4. **Allocate sufficient resources**: Minimum 4GB RAM, recommended 8GB+

## Next Steps

- Review [README.md](README.md) for architecture details
- Check [project_spec.md](project_spec.md) for technical specifications
- Review individual service READMEs in `backend/*/` directories

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review Docker and WSL2 logs
3. Consult official Docker documentation: https://docs.docker.com/
