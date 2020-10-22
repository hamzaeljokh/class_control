from django.contrib import admin

from .models import Formation, Etudiant, Seance, Absance

admin.site.register(Formation)
admin.site.register(Etudiant)
admin.site.register(Seance)
admin.site.register(Absance)

