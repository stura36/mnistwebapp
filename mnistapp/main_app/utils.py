from django.template.loader import render_to_string
from django.contrib import messages
from django.contrib import messages
from django.template.loader import render_to_string
from django.contrib.sites.shortcuts import get_current_site
from django.utils.http import urlsafe_base64_encode
from django.utils.encoding import force_bytes
from django.core.mail import EmailMessage
from django.contrib.auth.tokens import default_token_generator
from .tokens import account_activation_token
import torchvision.transforms as T
import io
import base64
import torch
from .architectures import ConvModel
import torch.nn.functional as F

def email_activation(request,user,to_email):

    mail_subject = "Activate your user account."
    message = render_to_string("main_app/activate_mail.html", {
    'user': user.username,
    'domain': get_current_site(request).domain,
    'uid': urlsafe_base64_encode(force_bytes(user.pk)),
    'token': account_activation_token.make_token(user),
    "protocol": 'https' if request.is_secure() else 'http'
    })
    email = EmailMessage(mail_subject, message, to=[to_email])
    
    if email.send():
        messages.success(request, f'Dear <b>{user}</b>, please go to you email <b>{to_email}</b> inbox and click on \
                received activation link to confirm and complete the registration. <b>Note:</b> Check your spam folder.')
    else:
        messages.error(request, f'Problem sending email to {to_email}, check if you typed it correctly.')


def email_reset(request,user,to_email):
    mail_subject = "Password reset."
    token = default_token_generator.make_token(user)
    
    message = render_to_string("main_app/reset_mail.html",{
    'user': user.username,
    'domain': get_current_site(request).domain,
    "uid": urlsafe_base64_encode(force_bytes(user.pk)),
    'token': token,
    "protocol": 'https' if request.is_secure() else 'http'
    })



    email = EmailMessage(mail_subject, message, to=[to_email])

    if email.send():
        messages.success(request, f'Dear <b>{user}</b>, please go to you email <b>{to_email}</b> inbox and click on \
                received activation link to confirm and complete the registration. <b>Note:</b> Check your spam folder.')
    else:
        messages.error(request, f'Problem sending email to {to_email}, check if you typed it correctly.')
        raise Exception()
def tensor_to_png_bytes(image_tensor):
    transform_to_pil = T.ToPILImage(mode = 'L')
    image_data = transform_to_pil(image_tensor).resize((200,200))
    buffer = io.BytesIO()
    image_data.save(buffer,format = 'PNG')
    
    image_data_bytes = buffer.getvalue()
    image_data_base64 = base64.b64encode(image_data_bytes).decode('utf-8')
    return image_data_base64

def load_setup_model(path_to_model = 'main_app/models/model.pth'):
    model = ConvModel()
    state_dict = torch.load(path_to_model, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def model_predict(model,image_input):
    with torch.no_grad():
        normalized_image = image_input / 255.
        normalized_image = normalized_image.unsqueeze(0) 
        output = model(normalized_image)
        output = F.softmax(output)
        prediction = torch.argmax(output, dim=1).item()
        return prediction
