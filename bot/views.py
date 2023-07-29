from django.shortcuts import render
from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large")
chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)


def chat(request):
    if request.method == 'POST':
        user_message = request.POST['message']
        bot_response = chatbot(user_message)[0]['generated_text']
        return JsonResponse({'bot_response': bot_response})