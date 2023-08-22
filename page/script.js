document.addEventListener("DOMContentLoaded", () => {
      const productContainers = document.querySelectorAll(".obj");

      productContainers.forEach(product => {
          const img = product.querySelector("img");
          const price = product.querySelector(".price");
          const altText = img.getAttribute("alt");
          const hiddenText = product.querySelector(".hidden-text").textContent;

          product.addEventListener("click", async () => {
              const elementContent = product.textContent; // Retrieve entire content
              const response = await fetch("/get_likes", {
                  method: "POST",
                  body: new URLSearchParams({
                      element_content: elementContent,
                      alt_text: altText,
                      hidden_text: hiddenText
                  }),
                  headers: {
                      "Content-Type": "application/x-www-form-urlencoded"
                  }
              });
              const data = await response.json();
              console.log(data);
          });
      });
  });

  
  
  
  