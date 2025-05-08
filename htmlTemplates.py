css = '''
<style>
.chat-container {
    display: flex;
    flex-direction: column;
}

.chat-row {
    display: flex;
    margin: 5px;
    width: 100%;
}

.row-reverse {
    flex-direction: row-reverse;
}

.chat-bubble {
    font-family: "Source Sans Pro", sans-serif, "Segoe UI", "Roboto", sans-serif;
    border: 1px solid transparent;
    padding-right: 8px;
    padding-left: 10px;
    margin-right: 4px ;
    max-width: 60%;
    /* Limita la altura mínima de las burbujas de chat */
    min-height: 30px; /* Puedes ajustar este valor según tus necesidades */
}

.ai-bubble {
    background: #0cd111;
    color: white;
    border-radius: 10px;
}

.human-bubble {
    background: #27b5f2;
    color: white;
    border-radius: 20px;
}

.chat-icon {
    border-radius: 5px;
}
</style>

'''

bot_template = '''
<div class="chat-row">
<div class="chat-icon">
            <img src="https://i.ibb.co/TgcShwr/AgaviNew.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
        </div>
    <div class="chat-bubble ai-bubble">
        {{MSG}}
    </div>
</div>
'''

user_template = '''
<div class="chat-row row-reverse">
<div class="chat-icon">
          <img src="https://i.ibb.co/mJ73WT9/Usuario.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">  
        </div>
    <div class="chat-bubble human-bubble">    
        {{MSG}}
    </div>
</div>
'''
