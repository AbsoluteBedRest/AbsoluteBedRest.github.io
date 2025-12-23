---
layout: page
title: Papers
permalink: /papers/
---

{% assign papers = site.posts | where_exp: "post", "post.categories contains 'papers'" %}

{% for post in papers %}
### [{{ post.title }}]({{ post.url }})
<span style="color: #888; font-size: 0.9em;">
{{ post.date | date: "%Y-%m-%d" }}
</span>

{{ post.excerpt }}

---
{% endfor %}