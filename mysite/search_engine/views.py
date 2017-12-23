from django.shortcuts import get_object_or_404, render
from django.http import HttpResponseRedirect, HttpResponse
from django.urls import reverse
from django.views import generic
from .models import Article


def index(request):
    return HttpResponse("Hello, world! You're at the search_engine index.")


# class DetailView(generic.DetailView):
#     model = Article
#     template_name = 'seach_engine/detail.html'

def detail(request, article_id):
    article = get_object_or_404(Article, pk=article_id)
    return render(request, 'search_engine/detail.html', {'article': article})
