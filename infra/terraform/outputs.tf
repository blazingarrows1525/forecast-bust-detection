output "dashboard_url" {
  description = "Where a duty forecaster opens the review queue."
  value       = "https://${aws_lb.main.dns_name}"
}

output "ecr_repository_url" {
  description = "Push target: docker tag fbd:0.1.0 <this>:<immutable-tag>"
  value       = aws_ecr_repository.app.repository_url
}

output "log_group" {
  value = aws_cloudwatch_log_group.app.name
}

output "genai_enabled" {
  description = "Whether this deployment granted Bedrock access at all."
  value       = var.enable_genai
}
