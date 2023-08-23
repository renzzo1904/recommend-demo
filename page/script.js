document.addEventListener("DOMContentLoaded", () => {
    const productContainers = document.querySelectorAll(".obj");

    productContainers.forEach(product => {
        const img = product.querySelector("img");
        const price = product.querySelector(".price");
        const altText = img.getAttribute("alt");
        const hiddenText = product.querySelector(".hidden-text").textContent;

        product.addEventListener("click", async () => {
            const elementContent = product.querySelector("h2").textContent; // Retrieve heading content
            const response = await fetch("/get_likes", {
                method: "POST",
                body: JSON.stringify({
                    element_content: elementContent,
                    alt_text: altText,
                    hidden_text: hiddenText
                }),
                headers: {
                    "Content-Type": "application/json"
                }
            });
            const data = await response.json();
            console.log(data);
        });
    });
});
  
  
  
  