from django.contrib.auth.forms import UserCreationForm, AuthenticationForm


class StyledMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            field.widget.attrs["class"] = "form-control"


class RegisterForm(StyledMixin, UserCreationForm):
    pass


class LoginForm(StyledMixin, AuthenticationForm):
    pass