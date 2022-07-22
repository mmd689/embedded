<h1 align="center">
پروژه‌ی پایانی درس سامانه‌های نهفته
</h1>
<h2 align="center">
  دانشكده مهندسی کامپيوتر
  
  دانشگاه صنعتی شریف      
  
  گزارش پروژه
</h2>


قطعات استفاده شده:
<ul>
  <li>بردبورد</li>
  <li>Raspberry pi</li>
  <li>دوربین برای تشخیص تصویر و فاصله از کامپیوتر (droid cam)</li>
  <li>سنسور HC-SR04 برای محاسبه‌ی فاصله‌ی کامپیوتر از فرد</li>
  <li>Male-to-female connector</li>
</ul>

توضیح فایل‌ها:

<ul dir='rtl'>
  <li>check_distance: تشخیص فاصله به کمک سنسور. پس از قرار دادن سنسور در بردبورد و اتصال آن به پین‌های مربوطه در Raspberry pi به کمک سیم، این بخش از کد فاصله‌ی فرد تا دوربین را اعلام می‌کند.</li>
  <li>check_glasses: تشخیص عینک داشتن یا نداشتن فرد. به کمک یک مدل از پیش train شده، تصویر  گرفته شده از دوربین را بررسی می‌کند و عینک داشتن یا نداشتنش را اعلام می‌کند.</li>
  <li>prepare_dataset: هنگام اضافه شدن هر کاربر، به کمک این کد تعدادی تصویر از او گرفته می‌شود که بعداً به کمک آنها چهره‌ی او قابل تشخیص باشد. </li>
  <li>train_model: تصاویر گرفته شده در بخش قبل، در این قسمت train می‌گردند. </li>
  <li>face_recognition: به کمک دیتاستی که از کاربران دستگاه دارد، تشخیص می‌دهد کاربر فعلی روبروی دوربین چه کسی است.</li>
  <li>main-project: بخش اصلی پروژه. به کمک تشخیص چهره، عینک داشتن یا نداشتن و فاصله، خروجی‌های مربوطه در این بخش تولید می‌گردند. </li>
  
</ul>