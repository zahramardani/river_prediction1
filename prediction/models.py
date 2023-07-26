from django.db import models

class Deployment(models.Model):
    DEPLOYMENT_TYPES = (
        (1, 'Time series'),
        (2, 'Others'),
    )

    name = models.CharField(max_length=100)
    deployment_type = models.IntegerField(choices=DEPLOYMENT_TYPES)
    application_field = models.CharField(max_length=100)

    def __str__(self):
        return self.name

