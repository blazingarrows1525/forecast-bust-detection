variable "region" {
  description = "AWS region. Must be one where the chosen Bedrock models are available."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  type    = string
  default = "demo"
}

variable "image_tag" {
  description = "Immutable image tag to deploy. Never 'latest' -- the ECR repo rejects mutable tags on purpose."
  type        = string
}

variable "certificate_arn" {
  description = "ACM certificate for the HTTPS listener."
  type        = string
}

variable "allowed_ingress_cidrs" {
  description = "Who may reach the load balancer. Default is deliberately NOT 0.0.0.0/0: this is an internal decision-support tool for duty forecasters, not a public site."
  type        = list(string)
  default     = ["10.0.0.0/8"]
}

variable "desired_count" {
  type    = number
  default = 2
}

variable "task_cpu" {
  description = "Fargate CPU units. The model is precomputed into SQLite, so serving is I/O bound and 512 is ample."
  type        = string
  default     = "512"
}

variable "task_memory" {
  type    = string
  default = "1024"
}

variable "log_retention_days" {
  type    = number
  default = 30
}

variable "enable_genai" {
  description = "Opt in to the Bedrock-backed assistant. Default false mirrors the application default (D-015): the offline path is the default everywhere, including in infrastructure."
  type        = bool
  default     = false
}

variable "bedrock_models" {
  description = "Claude model ids WITHOUT the anthropic. prefix; the ARN adds it."
  type        = list(string)
  default     = ["claude-opus-5", "claude-haiku-4-5"]
}

variable "bedrock_guardrail_id" {
  description = "Optional Bedrock Guardrail id. Complementary to the in-process guardrails, which enforce the domain invariants AWS cannot know about."
  type        = string
  default     = ""
}
