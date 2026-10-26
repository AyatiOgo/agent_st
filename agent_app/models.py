from django.db import models

# Create your models here.

from django.db import models


class HandbookModel(models.Model):
    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]
    title = models.CharField(max_length=255)
    file = models.FileField( upload_to="handbooks/")
    total_pages = models.PositiveIntegerField(default=0)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    error_message = models.TextField(blank=True )
    created_at = models.DateTimeField( auto_now_add=True )
    updated_at = models.DateTimeField( auto_now=True )
    def __str__(self):
        return self.title


class DocumentChunkModel(models.Model):
    handbook = models.ForeignKey( HandbookModel, on_delete=models.CASCADE, related_name="chunks")
    content = models.TextField()
    page_start = models.PositiveIntegerField()
    page_end = models.PositiveIntegerField()
    section = models.CharField(max_length=255, blank=True )
    chunk_index = models.PositiveIntegerField()
    embedding = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField( auto_now_add=True )
    
    def __str__(self):
        return (
            f"{self.handbook.title} - "
            f"Pages {self.page_start}-{self.page_end}"
        )