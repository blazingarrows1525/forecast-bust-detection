###############################################################################
# Forecast Bust Detection -- AWS deployment (SIH26079)
#
# NEVER APPLIED. This machine has no AWS credentials (D-015 verification
# boundary), so this configuration has been written and `terraform fmt`-shaped
# but never `plan`ned or `apply`ed against a real account. Treat it as a
# reviewed design, not as running infrastructure, and expect to fix things on
# first apply.
#
# Design notes worth defending:
#
#  * Fargate, not EC2 or EKS. LOGIC.md 14 forbids Kubernetes and the workload
#    is one stateless container serving a read-only SQLite file. Fargate is the
#    smallest thing that survives an availability-zone loss.
#  * The bulletin store is baked into the image, not mounted from EFS. It is
#    immutable per model version, 63 MB, and read-only -- so a shared
#    filesystem would add a failure mode and buy nothing. Redeploy to update.
#  * Bedrock access is opt-in and scoped to specific model ARNs. The task role
#    gets no bedrock:* wildcard, so a compromised container cannot invoke an
#    arbitrary model or reach another account's resources.
#  * GenAI stays off unless var.enable_genai is set, mirroring the application
#    default so infrastructure cannot silently switch it on.
###############################################################################

terraform {
  required_version = ">= 1.6.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
  }
}

provider "aws" {
  region = var.region
  default_tags {
    tags = {
      Project = "forecast-bust-detection"
      PS      = "SIH26079"
      Owner   = "MoES-IMD-decision-support"
      ManagedBy = "terraform"
    }
  }
}

data "aws_caller_identity" "current" {}

locals {
  name = "fbd-${var.environment}"
  # Bedrock model ids carry an "anthropic." prefix; the ARN is per-region.
  bedrock_model_arns = [
    for m in var.bedrock_models :
    "arn:aws:bedrock:${var.region}::foundation-model/anthropic.${m}"
  ]
}

###############################################################################
# Networking
###############################################################################
resource "aws_vpc" "main" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true
  tags                 = { Name = local.name }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
}

# Two AZs. One is a single point of failure; three costs more NAT than a
# decision-support service for 34 subdivisions can justify.
resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true
  tags                    = { Name = "${local.name}-public-${count.index}" }
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

###############################################################################
# Security groups
###############################################################################
resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "Ingress to the load balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTPS from the allowed CIDRs"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = var.allowed_ingress_cidrs
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "service" {
  name        = "${local.name}-service"
  description = "Task ingress from the ALB only"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "App port, ALB only"
    from_port       = 8912
    to_port         = 8912
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "Outbound for ECR pulls, logs, and Bedrock when enabled"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

###############################################################################
# Container registry and logs
###############################################################################
resource "aws_ecr_repository" "app" {
  name                 = local.name
  image_tag_mutability = "IMMUTABLE" # a deployed tag must mean one image
  image_scanning_configuration {
    scan_on_push = true
  }
  encryption_configuration {
    encryption_type = "AES256"
  }
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${local.name}"
  retention_in_days = var.log_retention_days
}

###############################################################################
# IAM
###############################################################################
data "aws_iam_policy_document" "task_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "execution" {
  name               = "${local.name}-execution"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
}

resource "aws_iam_role_policy_attachment" "execution" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# The application role. Note what is absent: no S3, no DynamoDB, no
# bedrock:* wildcard. The container serves a baked-in SQLite file and, when
# explicitly enabled, invokes a named list of models. Nothing else.
resource "aws_iam_role" "task" {
  name               = "${local.name}-task"
  assume_role_policy = data.aws_iam_policy_document.task_assume.json
}

data "aws_iam_policy_document" "bedrock" {
  count = var.enable_genai ? 1 : 0

  statement {
    sid       = "InvokeNamedClaudeModelsOnly"
    effect    = "Allow"
    actions   = ["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"]
    resources = local.bedrock_model_arns
  }

  dynamic "statement" {
    for_each = var.bedrock_guardrail_id == "" ? [] : [1]
    content {
      sid     = "ApplyGuardrail"
      effect  = "Allow"
      actions = ["bedrock:ApplyGuardrail"]
      resources = [
        "arn:aws:bedrock:${var.region}:${data.aws_caller_identity.current.account_id}:guardrail/${var.bedrock_guardrail_id}"
      ]
    }
  }
}

resource "aws_iam_role_policy" "bedrock" {
  count  = var.enable_genai ? 1 : 0
  name   = "${local.name}-bedrock"
  role   = aws_iam_role.task.id
  policy = data.aws_iam_policy_document.bedrock[0].json
}

###############################################################################
# ECS service
###############################################################################
resource "aws_ecs_cluster" "main" {
  name = local.name
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

resource "aws_ecs_task_definition" "app" {
  family                   = local.name
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = aws_iam_role.execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "fbd"
      image     = "${aws_ecr_repository.app.repository_url}:${var.image_tag}"
      essential = true
      portMappings = [{ containerPort = 8912, protocol = "tcp" }]
      readonlyRootFilesystem = true
      environment = [
        # Mirrors the application default. Infrastructure must not be the
        # thing that switches the GenAI layer on behind the operator's back.
        { name = "FBD_GENAI_ENABLED", value = var.enable_genai ? "1" : "0" },
        { name = "FBD_GENAI_TOOLS", value = var.enable_genai ? "1" : "0" },
        { name = "FBD_GENAI_RAG", value = var.enable_genai ? "1" : "0" },
        { name = "AWS_REGION", value = var.region },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "fbd"
        }
      }
      healthCheck = {
        command     = ["CMD-SHELL", "python -c \"import urllib.request;urllib.request.urlopen('http://127.0.0.1:8912/api/health',timeout=4)\" || exit 1"]
        interval    = 30
        timeout     = 5
        retries     = 3
        startPeriod = 20
      }
    }
  ])
}

resource "aws_lb" "main" {
  name               = local.name
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
  drop_invalid_header_fields = true
}

resource "aws_lb_target_group" "app" {
  name        = local.name
  port        = 8912
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    path                = "/api/health"
    matcher             = "200"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

resource "aws_ecs_service" "app" {
  name            = local.name
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.service.id]
    assign_public_ip = true
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "fbd"
    container_port   = 8912
  }

  depends_on = [aws_lb_listener.https]
}
