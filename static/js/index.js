document
  .querySelectorAll(".publication-image img")
  .forEach((image) => {
    image.addEventListener("error", () => {
      image.closest(".publication-image").classList.add("is-missing");
    });
  });

const carousel = document.querySelector("[data-carousel]");
const previousButton = document.querySelector("[data-carousel-prev]");
const nextButton = document.querySelector("[data-carousel-next]");
const dots = Array.from(document.querySelectorAll(".carousel-dot"));

if (carousel && previousButton && nextButton && dots.length > 0) {
  const slides = Array.from(carousel.querySelectorAll(".carousel-item"));
  let activeIndex = 0;

  const setActiveDot = (index) => {
    dots.forEach((dot, dotIndex) => {
      dot.classList.toggle("is-active", dotIndex === index);
    });
  };

  const showSlide = (index) => {
    activeIndex = (index + slides.length) % slides.length;
    slides[activeIndex].scrollIntoView({
      behavior: "smooth",
      block: "nearest",
      inline: "start",
    });
    setActiveDot(activeIndex);
  };

  previousButton.addEventListener("click", () => showSlide(activeIndex - 1));
  nextButton.addEventListener("click", () => showSlide(activeIndex + 1));

  dots.forEach((dot, index) => {
    dot.addEventListener("click", () => showSlide(index));
  });

  carousel.addEventListener("scroll", () => {
    const nextIndex = Math.round(carousel.scrollLeft / carousel.clientWidth);
    if (nextIndex !== activeIndex && nextIndex >= 0 && nextIndex < slides.length) {
      activeIndex = nextIndex;
      setActiveDot(activeIndex);
    }
  });
}
