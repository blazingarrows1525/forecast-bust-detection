# Terraform deployment

> **Status: never applied, and never even validated.** This repository was
> built on a machine with no AWS credentials *and no terraform binary*, so
> nothing here has run `terraform validate`, `plan`, or `apply` (DECISIONS.md
> D-015). The HCL has not been machine-checked at all -- assume syntax errors
> as well as design errors on first use. Run `terraform fmt -check` and
> `terraform validate` before anything else. Do not present this as running
> infrastructure.

## What it creates

VPC across two AZs, an ALB terminating TLS 1.3, an ECS Fargate service running
the `fbd` image, an immutable-tag ECR repository with scan-on-push, and
CloudWatch logs. Optionally, a tightly scoped Bedrock invoke policy.

## Deploy

```bash
docker build -t fbd:0.1.0 .
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com
docker tag fbd:0.1.0 <account>.dkr.ecr.us-east-1.amazonaws.com/fbd-demo:0.1.0
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/fbd-demo:0.1.0

terraform init
terraform plan  -var image_tag=0.1.0 -var certificate_arn=arn:aws:acm:...
terraform apply -var image_tag=0.1.0 -var certificate_arn=arn:aws:acm:...
```

## Deliberate choices a reviewer may challenge

| Choice | Why |
|---|---|
| Fargate, not EKS | LOGIC.md 14 forbids Kubernetes. One stateless container does not need an orchestrator. |
| SQLite baked into the image, no EFS | The store is immutable per model version, 63 MB, read-only. A shared filesystem adds a failure mode and buys nothing. |
| `image_tag` required, ECR tags immutable | A deployed tag must identify exactly one image. `latest` cannot. |
| Ingress defaults to RFC1918, not `0.0.0.0/0` | This is an internal tool for duty forecasters. Public exposure should be a deliberate override. |
| `enable_genai = false` by default | Infrastructure must not be what silently switches on network egress to an LLM (D-015). |
| Bedrock policy names model ARNs | No `bedrock:*`. A compromised container cannot invoke arbitrary models. |
| `readonlyRootFilesystem = true` | The app writes nothing at runtime. |
