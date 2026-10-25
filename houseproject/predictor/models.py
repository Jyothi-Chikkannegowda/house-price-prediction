from django.db import models
from django.contrib.auth.models import User


class Prediction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="predictions")
    location = models.CharField(max_length=100)
    sqft = models.FloatField()
    bhk = models.IntegerField()
    bath = models.IntegerField()
    price_lakhs = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    @property
    def price_text(self):
        if self.price_lakhs >= 100:
            return f"₹{self.price_lakhs / 100:.2f} Cr"
        return f"₹{self.price_lakhs:.2f} Lakhs"

    def __str__(self):
        return f"{self.user.username} - {self.location} - {self.price_text}"