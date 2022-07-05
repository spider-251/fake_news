from django.db import models

# Create your models here.
class News(models.Model):
    input = models.CharField(max_length=250)
    search_date = models.DateTimeField()
    lg_prob = models.DecimalField(max_digits=4,decimal_places=2)
    fk_prob = models.DecimalField(max_digits=4,decimal_places=2)
    label = models.CharField(max_length=10)
    def __str__(self):
        return self.input

