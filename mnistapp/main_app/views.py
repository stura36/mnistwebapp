from time import sleep
from django.shortcuts import render

# Create your views here.
from django.shortcuts import  render, redirect, reverse
from .forms import NewUserForm
from django.contrib.auth import login, authenticate, logout, get_user_model
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm,PasswordResetForm
from django.utils.encoding import force_str
from django.utils.http import urlsafe_base64_decode
from .tokens import account_activation_token


from torchvision import datasets, transforms
import random

from .utils import (email_activation,
		    email_reset,
			tensor_to_png_bytes,
			load_setup_model,
			model_predict)

def register_request(request):
	if request.user.is_authenticated:
		logout(request)
	if request.method == "POST":
		form = NewUserForm(request.POST)
		if form.is_valid():
			user = form.save(commit = False)
			user.is_active = False
			user.save()

			email_activation(request,user,form.cleaned_data.get('email'))
			return redirect(reverse("login_request"))
		else:
			messages.error(request, "Unsuccessful registration. Invalid information.")
	form = NewUserForm()
	return render (request=request, template_name="main_app/register.html", context={"register_form":form})

def login_request(request):
	if request.user.is_authenticated:
		logout(request)
	if request.method == "POST":
		form = AuthenticationForm(request, data=request.POST)
		if form.is_valid():
			username = form.cleaned_data.get('username')
			password = form.cleaned_data.get('password')
			user = authenticate(username=username, password=password)
			if user is not None:
				login(request, user)
				return redirect(reverse("index"))
			else:
				messages.error(request,"Invalid username or password.")
				
		else:
			messages.error(request,"Invalid username or password.")
	form = AuthenticationForm()
	return render(request=request, template_name="main_app/login.html", context={"login_form":form})

def logout_request(request):
	logout(request)
	return redirect(reverse("login_request"))


def activate(request,uidb64, token):
	User = get_user_model()
	try:
		uid = force_str(urlsafe_base64_decode(uidb64))
		user = User.objects.get(pk=uid)
	except:
		user = None
	context = {}

	if user is not None and account_activation_token.check_token(user, token):
		user.is_active = True
		user.save()
		context["message"] = "Your account is now activated, you can login."
	else:
		context["message"] = "Link is invalid!"
	return render(request = request,template_name="main_app/mail_activated.html",context = context)
	

def password_reset_request(request):
	if request.user.is_authenticated:
		logout(request)
	if request.method == "POST":
		password_reset_form = PasswordResetForm(request.POST)
		if password_reset_form.is_valid():
			data = password_reset_form.cleaned_data['email']
			try:
				email_reset(request,request.user,data)
				return redirect ("/password_reset/done/")
			except:
				messages.error(request,"Mail not registered.")
				
	password_reset_form = PasswordResetForm()
	return render(request=request, template_name="main_app/password/password_reset.html", context={"password_reset_form":password_reset_form})




def index(request):
	if request.user.is_authenticated:
		return classify_rand_img(request)
	else:
		return redirect(reverse("login_request"))



def classify_rand_img(request):
	context = {}
	if request.method == 'POST':

		model = load_setup_model()
		transform = transforms.Compose([
			transforms.ToTensor()
		])
		dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
		index = random.randint(0, len(dataset) - 1)

		image, label = dataset[index]
		prediction = model_predict(model,image)
		image_data_base64 = tensor_to_png_bytes(image)

		context = {'image': image_data_base64,'prediction': prediction}
		
	return render(request=request, template_name="main_app/classify.html", context = context)